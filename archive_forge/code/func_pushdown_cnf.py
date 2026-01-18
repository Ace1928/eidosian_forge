from sqlglot import exp
from sqlglot.optimizer.normalize import normalized
from sqlglot.optimizer.scope import build_scope, find_in_scope
from sqlglot.optimizer.simplify import simplify
def pushdown_cnf(predicates, sources, scope_ref_count, join_index=None):
    """
    If the predicates are in CNF like form, we can simply replace each block in the parent.
    """
    join_index = join_index or {}
    for predicate in predicates:
        for node in nodes_for_predicate(predicate, sources, scope_ref_count).values():
            if isinstance(node, exp.Join):
                name = node.alias_or_name
                predicate_tables = exp.column_table_names(predicate, name)
                this_index = join_index[name]
                if all((join_index.get(table, -1) < this_index for table in predicate_tables)):
                    predicate.replace(exp.true())
                    node.on(predicate, copy=False)
                    break
            if isinstance(node, exp.Select):
                predicate.replace(exp.true())
                inner_predicate = replace_aliases(node, predicate)
                if find_in_scope(inner_predicate, exp.AggFunc):
                    node.having(inner_predicate, copy=False)
                else:
                    node.where(inner_predicate, copy=False)