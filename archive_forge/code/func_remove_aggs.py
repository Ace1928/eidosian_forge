from sqlglot import exp
from sqlglot.helper import name_sequence
from sqlglot.optimizer.scope import ScopeType, traverse_scope
def remove_aggs(node):
    if isinstance(node, exp.Count):
        return exp.Literal.number(0)
    elif isinstance(node, exp.AggFunc):
        return exp.null()
    return node