from sys import intern
def node_init(self, node):
    assert not self.node_is_on_list(node)
    self.node_set_next(node, node)
    self.node_set_prev(node, node)