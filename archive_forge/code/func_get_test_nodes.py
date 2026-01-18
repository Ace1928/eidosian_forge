important if you want to refactor a parser tree.
import re
from typing import Tuple
from parso.tree import Node, BaseNode, Leaf, ErrorNode, ErrorLeaf, search_ancestor  # noqa
from parso.python.prefix import split_prefix
from parso.utils import split_lines
def get_test_nodes(self):
    """
        E.g. returns all the `test` nodes that are named as x, below:

            if x:
                pass
            elif x:
                pass
        """
    for i, c in enumerate(self.children):
        if c in ('elif', 'if'):
            yield self.children[i + 1]