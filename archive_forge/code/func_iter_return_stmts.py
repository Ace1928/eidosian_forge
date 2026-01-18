important if you want to refactor a parser tree.
import re
from typing import Tuple
from parso.tree import Node, BaseNode, Leaf, ErrorNode, ErrorLeaf, search_ancestor  # noqa
from parso.python.prefix import split_prefix
from parso.utils import split_lines
def iter_return_stmts(self):
    """
        Returns a generator of `return_stmt`.
        """

    def scan(children):
        for element in children:
            if element.type == 'return_stmt' or (element.type == 'keyword' and element.value == 'return'):
                yield element
            if element.type in _RETURN_STMT_CONTAINERS:
                yield from scan(element.children)
    return scan(self.children)