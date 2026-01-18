from __future__ import annotations
import typing as t
from sqlglot import exp
from sqlglot.helper import tsort
def reorder_joins(expression):
    """
    Reorder joins by topological sort order based on predicate references.
    """
    for from_ in expression.find_all(exp.From):
        parent = from_.parent
        joins = {join.alias_or_name: join for join in parent.args.get('joins', [])}
        dag = {name: other_table_names(join) for name, join in joins.items()}
        parent.set('joins', [joins[name] for name in tsort(dag) if name != from_.alias_or_name and name in joins])
    return expression