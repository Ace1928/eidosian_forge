from collections import defaultdict
from sqlglot import expressions as exp
from sqlglot.helper import find_new_name
from sqlglot.optimizer.scope import Scope, traverse_scope
def merge_subqueries(expression, leave_tables_isolated=False):
    """
    Rewrite sqlglot AST to merge derived tables into the outer query.

    This also merges CTEs if they are selected from only once.

    Example:
        >>> import sqlglot
        >>> expression = sqlglot.parse_one("SELECT a FROM (SELECT x.a FROM x) CROSS JOIN y")
        >>> merge_subqueries(expression).sql()
        'SELECT x.a FROM x CROSS JOIN y'

    If `leave_tables_isolated` is True, this will not merge inner queries into outer
    queries if it would result in multiple table selects in a single query:
        >>> expression = sqlglot.parse_one("SELECT a FROM (SELECT x.a FROM x) CROSS JOIN y")
        >>> merge_subqueries(expression, leave_tables_isolated=True).sql()
        'SELECT a FROM (SELECT x.a FROM x) CROSS JOIN y'

    Inspired by https://dev.mysql.com/doc/refman/8.0/en/derived-table-optimization.html

    Args:
        expression (sqlglot.Expression): expression to optimize
        leave_tables_isolated (bool):
    Returns:
        sqlglot.Expression: optimized expression
    """
    expression = merge_ctes(expression, leave_tables_isolated)
    expression = merge_derived_tables(expression, leave_tables_isolated)
    return expression