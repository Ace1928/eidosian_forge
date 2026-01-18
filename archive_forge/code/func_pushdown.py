from sqlglot import exp
from sqlglot.optimizer.normalize import normalized
from sqlglot.optimizer.scope import build_scope, find_in_scope
from sqlglot.optimizer.simplify import simplify
def pushdown(condition, sources, scope_ref_count, dialect, join_index=None):
    if not condition:
        return
    condition = condition.replace(simplify(condition, dialect=dialect))
    cnf_like = normalized(condition) or not normalized(condition, dnf=True)
    predicates = list(condition.flatten() if isinstance(condition, exp.And if cnf_like else exp.Or) else [condition])
    if cnf_like:
        pushdown_cnf(predicates, sources, scope_ref_count, join_index=join_index)
    else:
        pushdown_dnf(predicates, sources, scope_ref_count)