important if you want to refactor a parser tree.
import re
from typing import Tuple
from parso.tree import Node, BaseNode, Leaf, ErrorNode, ErrorLeaf, search_ancestor  # noqa
from parso.python.prefix import split_prefix
from parso.utils import split_lines
def is_generator(self):
    """
        :return bool: Checks if a function is a generator or not.
        """
    return next(self.iter_yield_exprs(), None) is not None