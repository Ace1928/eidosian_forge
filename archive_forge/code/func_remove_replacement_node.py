from functools import total_ordering
from django.db.migrations.state import ProjectState
from .exceptions import CircularDependencyError, NodeNotFoundError
def remove_replacement_node(self, replacement, replaced):
    """
        The inverse operation to `remove_replaced_nodes`. Almost. Remove the
        replacement node `replacement` and remap its child nodes to `replaced`
        - the list of nodes it would have replaced. Don't remap its parent
        nodes as they are expected to be correct already.
        """
    self.nodes.pop(replacement, None)
    try:
        replacement_node = self.node_map.pop(replacement)
    except KeyError as err:
        raise NodeNotFoundError('Unable to remove replacement node %r. It was either never added to the migration graph, or has been removed already.' % (replacement,), replacement) from err
    replaced_nodes = set()
    replaced_nodes_parents = set()
    for key in replaced:
        replaced_node = self.node_map.get(key)
        if replaced_node:
            replaced_nodes.add(replaced_node)
            replaced_nodes_parents |= replaced_node.parents
    replaced_nodes -= replaced_nodes_parents
    for child in replacement_node.children:
        child.parents.remove(replacement_node)
        for replaced_node in replaced_nodes:
            replaced_node.add_child(child)
            child.add_parent(replaced_node)
    for parent in replacement_node.parents:
        parent.children.remove(replacement_node)