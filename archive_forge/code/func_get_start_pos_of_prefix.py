important if you want to refactor a parser tree.
import re
from typing import Tuple
from parso.tree import Node, BaseNode, Leaf, ErrorNode, ErrorLeaf, search_ancestor  # noqa
from parso.python.prefix import split_prefix
from parso.utils import split_lines
def get_start_pos_of_prefix(self):
    """
        Basically calls :py:meth:`parso.tree.NodeOrLeaf.get_start_pos_of_prefix`.
        """
    previous_leaf = self.get_previous_leaf()
    if previous_leaf is not None and previous_leaf.type == 'error_leaf' and (previous_leaf.token_type in ('INDENT', 'DEDENT', 'ERROR_DEDENT')):
        previous_leaf = previous_leaf.get_previous_leaf()
    if previous_leaf is None:
        lines = split_lines(self.prefix)
        return (self.line - len(lines) + 1, 0)
    return previous_leaf.end_pos