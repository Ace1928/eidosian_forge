from sqlglot import exp
from sqlglot.optimizer.normalize import normalized
from sqlglot.optimizer.scope import build_scope, find_in_scope
from sqlglot.optimizer.simplify import simplify
def pushdown_dnf(predicates, sources, scope_ref_count):
    """
    If the predicates are in DNF form, we can only push down conditions that are in all blocks.
    Additionally, we can't remove predicates from their original form.
    """
    pushdown_tables = set()
    for a in predicates:
        a_tables = exp.column_table_names(a)
        for b in predicates:
            a_tables &= exp.column_table_names(b)
        pushdown_tables.update(a_tables)
    conditions = {}
    for table in sorted(pushdown_tables):
        for predicate in predicates:
            nodes = nodes_for_predicate(predicate, sources, scope_ref_count)
            if table not in nodes:
                continue
            conditions[table] = exp.or_(conditions[table], predicate) if table in conditions else predicate
        for name, node in nodes.items():
            if name not in conditions:
                continue
            predicate = conditions[name]
            if isinstance(node, exp.Join):
                node.on(predicate, copy=False)
            elif isinstance(node, exp.Select):
                inner_predicate = replace_aliases(node, predicate)
                if find_in_scope(inner_predicate, exp.AggFunc):
                    node.having(inner_predicate, copy=False)
                else:
                    node.where(inner_predicate, copy=False)