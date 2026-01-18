from sqlglot import exp
from sqlglot.optimizer.normalize import normalized
from sqlglot.optimizer.scope import build_scope, find_in_scope
from sqlglot.optimizer.simplify import simplify
def nodes_for_predicate(predicate, sources, scope_ref_count):
    nodes = {}
    tables = exp.column_table_names(predicate)
    where_condition = isinstance(predicate.find_ancestor(exp.Join, exp.Where), exp.Where)
    for table in sorted(tables):
        node, source = sources.get(table) or (None, None)
        if node and where_condition:
            node = node.find_ancestor(exp.Join, exp.From)
        if isinstance(node, exp.From) and (not isinstance(source, exp.Table)):
            with_ = source.parent.expression.args.get('with')
            if with_ and with_.recursive:
                return {}
            node = source.expression
        if isinstance(node, exp.Join):
            if node.side and node.side != 'RIGHT':
                return {}
            nodes[table] = node
        elif isinstance(node, exp.Select) and len(tables) == 1:
            has_window_expression = any((select for select in node.selects if select.find(exp.Window)))
            if not node.args.get('group') and scope_ref_count[id(source)] < 2 and (not has_window_expression):
                nodes[table] = node
    return nodes