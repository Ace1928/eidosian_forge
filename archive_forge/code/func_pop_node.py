from . import errors
from . import graph as _mod_graph
from . import revision as _mod_revision
def pop_node(node_name_stack_pop=node_name_stack.pop, node_merge_depth_stack_pop=node_merge_depth_stack.pop, first_child_stack_pop=self._first_child_stack.pop, left_subtree_pushed_stack_pop=left_subtree_pushed_stack.pop, pending_parents_stack_pop=pending_parents_stack.pop, original_graph=self._original_graph, revnos=self._revnos, completed_node_names_add=self._completed_node_names.add, scheduled_nodes_append=scheduled_nodes.append, revno_to_branch_count=self._revno_to_branch_count):
    """Pop the top node off the stack

            The node is appended to the sorted output.
            """
    node_name = node_name_stack_pop()
    merge_depth = node_merge_depth_stack_pop()
    first_child = first_child_stack_pop()
    left_subtree_pushed_stack_pop()
    pending_parents_stack_pop()
    parents = original_graph[node_name]
    parent_revno = None
    if parents:
        try:
            parent_revno = revnos[parents[0]][0]
        except KeyError:
            pass
    if parent_revno is not None:
        if not first_child:
            base_revno = parent_revno[0]
            branch_count = revno_to_branch_count.get(base_revno, 0)
            branch_count += 1
            revno_to_branch_count[base_revno] = branch_count
            revno = (parent_revno[0], branch_count, 1)
        else:
            revno = parent_revno[:-1] + (parent_revno[-1] + 1,)
    else:
        root_count = revno_to_branch_count.get(0, -1)
        root_count += 1
        if root_count:
            revno = (0, root_count, 1)
        else:
            revno = (1,)
        revno_to_branch_count[0] = root_count
    revnos[node_name][0] = revno
    completed_node_names_add(node_name)
    scheduled_nodes_append((node_name, merge_depth, revno))
    return node_name