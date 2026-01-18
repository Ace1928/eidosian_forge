from sys import intern
def node_is_on_list(self, node):
    """Returns True if this node is on *some* list.

        A node is not on any list if it is linked to itself, or if it
        does not have the next and/prev attributes at all.
        """
    next = self.node_next(node)
    if next == node or next is None:
        assert self.node_prev(node) is next
        return False
    return True