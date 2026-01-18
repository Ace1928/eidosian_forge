from sys import intern
def node_prev(self, node):
    try:
        return getattr(node, self.prev_name)
    except AttributeError:
        return None