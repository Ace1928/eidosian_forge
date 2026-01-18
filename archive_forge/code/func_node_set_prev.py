from sys import intern
def node_set_prev(self, node, prev):
    setattr(node, self.prev_name, prev)