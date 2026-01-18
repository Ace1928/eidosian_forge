important if you want to refactor a parser tree.
import re
from typing import Tuple
from parso.tree import Node, BaseNode, Leaf, ErrorNode, ErrorLeaf, search_ancestor  # noqa
from parso.python.prefix import split_prefix
from parso.utils import split_lines
def is_node_after_else(self, node):
    """
        Checks if a node is defined after `else`.
        """
    for c in self.children:
        if c == 'else':
            if node.start_pos > c.start_pos:
                return True
    else:
        return False