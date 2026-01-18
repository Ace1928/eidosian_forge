from __future__ import annotations
import typing as t
from sqlglot import expressions as exp
from sqlglot.helper import find_new_name, name_sequence
def eliminate_qualify(expression: exp.Expression) -> exp.Expression:
    """
    Convert SELECT statements that contain the QUALIFY clause into subqueries, filtered equivalently.

    The idea behind this transformation can be seen in Snowflake's documentation for QUALIFY:
    https://docs.snowflake.com/en/sql-reference/constructs/qualify

    Some dialects don't support window functions in the WHERE clause, so we need to include them as
    projections in the subquery, in order to refer to them in the outer filter using aliases. Also,
    if a column is referenced in the QUALIFY clause but is not selected, we need to include it too,
    otherwise we won't be able to refer to it in the outer query's WHERE clause. Finally, if a
    newly aliased projection is referenced in the QUALIFY clause, it will be replaced by the
    corresponding expression to avoid creating invalid column references.
    """
    if isinstance(expression, exp.Select) and expression.args.get('qualify'):
        taken = set(expression.named_selects)
        for select in expression.selects:
            if not select.alias_or_name:
                alias = find_new_name(taken, '_c')
                select.replace(exp.alias_(select, alias))
                taken.add(alias)
        outer_selects = exp.select(*[select.alias_or_name for select in expression.selects])
        qualify_filters = expression.args['qualify'].pop().this
        expression_by_alias = {select.alias: select.this for select in expression.selects if isinstance(select, exp.Alias)}
        select_candidates = exp.Window if expression.is_star else (exp.Window, exp.Column)
        for select_candidate in qualify_filters.find_all(select_candidates):
            if isinstance(select_candidate, exp.Window):
                if expression_by_alias:
                    for column in select_candidate.find_all(exp.Column):
                        expr = expression_by_alias.get(column.name)
                        if expr:
                            column.replace(expr)
                alias = find_new_name(expression.named_selects, '_w')
                expression.select(exp.alias_(select_candidate, alias), copy=False)
                column = exp.column(alias)
                if isinstance(select_candidate.parent, exp.Qualify):
                    qualify_filters = column
                else:
                    select_candidate.replace(column)
            elif select_candidate.name not in expression.named_selects:
                expression.select(select_candidate.copy(), copy=False)
        return outer_selects.from_(expression.subquery(alias='_t', copy=False), copy=False).where(qualify_filters, copy=False)
    return expression