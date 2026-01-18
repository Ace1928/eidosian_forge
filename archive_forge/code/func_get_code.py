important if you want to refactor a parser tree.
import re
from typing import Tuple
from parso.tree import Node, BaseNode, Leaf, ErrorNode, ErrorLeaf, search_ancestor  # noqa
from parso.python.prefix import split_prefix
from parso.utils import split_lines
def get_code(self, include_prefix=True, include_comma=True):
    """
        Like all the other get_code functions, but includes the param
        `include_comma`.

        :param include_comma bool: If enabled includes the comma in the string output.
        """
    if include_comma:
        return super().get_code(include_prefix)
    children = self.children
    if children[-1] == ',':
        children = children[:-1]
    return self._get_code_for_children(children, include_prefix=include_prefix)