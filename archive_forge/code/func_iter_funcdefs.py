important if you want to refactor a parser tree.
import re
from typing import Tuple
from parso.tree import Node, BaseNode, Leaf, ErrorNode, ErrorLeaf, search_ancestor  # noqa
from parso.python.prefix import split_prefix
from parso.utils import split_lines
def iter_funcdefs(self):
    """
        Returns a generator of `funcdef` nodes.
        """
    return self._search_in_scope('funcdef')