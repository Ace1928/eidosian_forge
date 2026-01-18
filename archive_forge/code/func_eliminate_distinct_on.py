from __future__ import annotations
import typing as t
from sqlglot import expressions as exp
from sqlglot.helper import find_new_name, name_sequence
def eliminate_distinct_on(expression: exp.Expression) -> exp.Expression:
    """
    Convert SELECT DISTINCT ON statements to a subquery with a window function.

    This is useful for dialects that don't support SELECT DISTINCT ON but support window functions.

    Args:
        expression: the expression that will be transformed.

    Returns:
        The transformed expression.
    """
    if isinstance(expression, exp.Select) and expression.args.get('distinct') and expression.args['distinct'].args.get('on') and isinstance(expression.args['distinct'].args['on'], exp.Tuple):
        distinct_cols = expression.args['distinct'].pop().args['on'].expressions
        outer_selects = expression.selects
        row_number = find_new_name(expression.named_selects, '_row_number')
        window = exp.Window(this=exp.RowNumber(), partition_by=distinct_cols)
        order = expression.args.get('order')
        if order:
            window.set('order', order.pop())
        else:
            window.set('order', exp.Order(expressions=[c.copy() for c in distinct_cols]))
        window = exp.alias_(window, row_number)
        expression.select(window, copy=False)
        return exp.select(*outer_selects, copy=False).from_(expression.subquery('_t', copy=False), copy=False).where(exp.column(row_number).eq(1), copy=False)
    return expression