important if you want to refactor a parser tree.
import re
from typing import Tuple
from parso.tree import Node, BaseNode, Leaf, ErrorNode, ErrorLeaf, search_ancestor  # noqa
from parso.python.prefix import split_prefix
from parso.utils import split_lines
@property
def position_index(self):
    """
        Property for the positional index of a paramter.
        """
    index = self.parent.children.index(self)
    try:
        keyword_only_index = self.parent.children.index('*')
        if index > keyword_only_index:
            index -= 2
    except ValueError:
        pass
    try:
        keyword_only_index = self.parent.children.index('/')
        if index > keyword_only_index:
            index -= 2
    except ValueError:
        pass
    return index - 1