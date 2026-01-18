import re
import docutils
from docutils import nodes, writers, languages
def first_child(self, node):
    first = isinstance(node.parent[0], nodes.label)
    for child in node.parent.children[first:]:
        if isinstance(child, nodes.Invisible):
            continue
        if child is node:
            return 1
        break
    return 0