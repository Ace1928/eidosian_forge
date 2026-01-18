from sys import intern
def node_unlink(self, node):
    if not self.node_is_on_list(node):
        return
    prev = self.node_prev(node)
    next = self.node_next(node)
    self.node_set_next(prev, next)
    self.node_set_prev(next, prev)
    self.node_set_next(node, node)
    self.node_set_prev(node, node)