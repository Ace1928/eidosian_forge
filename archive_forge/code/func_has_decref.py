from collections import defaultdict
def has_decref(self, node):
    return 'decref' in self.nodes[node]