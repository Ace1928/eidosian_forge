important if you want to refactor a parser tree.
import re
from typing import Tuple
from parso.tree import Node, BaseNode, Leaf, ErrorNode, ErrorLeaf, search_ancestor  # noqa
from parso.python.prefix import split_prefix
from parso.utils import split_lines
def get_super_arglist(self):
    """
        Returns the `arglist` node that defines the super classes. It returns
        None if there are no arguments.
        """
    if self.children[2] != '(':
        return None
    elif self.children[3] == ')':
        return None
    else:
        return self.children[3]