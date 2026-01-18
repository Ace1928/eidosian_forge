important if you want to refactor a parser tree.
import re
from typing import Tuple
from parso.tree import Node, BaseNode, Leaf, ErrorNode, ErrorLeaf, search_ancestor  # noqa
from parso.python.prefix import split_prefix
from parso.utils import split_lines
def get_from_names(self):
    for n in self.children[1:]:
        if n not in ('.', '...'):
            break
    if n.type == 'dotted_name':
        return n.children[::2]
    elif n == 'import':
        return []
    else:
        return [n]