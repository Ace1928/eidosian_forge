important if you want to refactor a parser tree.
import re
from typing import Tuple
from parso.tree import Node, BaseNode, Leaf, ErrorNode, ErrorLeaf, search_ancestor  # noqa
from parso.python.prefix import split_prefix
from parso.utils import split_lines
def get_rhs(self):
    """Returns the right-hand-side of the equals."""
    node = self.children[-1]
    if node.type == 'annassign':
        if len(node.children) == 4:
            node = node.children[3]
        else:
            node = node.children[1]
    return node