important if you want to refactor a parser tree.
import re
from typing import Tuple
from parso.tree import Node, BaseNode, Leaf, ErrorNode, ErrorLeaf, search_ancestor  # noqa
from parso.python.prefix import split_prefix
from parso.utils import split_lines
def get_used_names(self):
    """
        Returns all the :class:`Name` leafs that exist in this module. This
        includes both definitions and references of names.
        """
    if self._used_names is None:
        dct = {}

        def recurse(node):
            try:
                children = node.children
            except AttributeError:
                if node.type == 'name':
                    arr = dct.setdefault(node.value, [])
                    arr.append(node)
            else:
                for child in children:
                    recurse(child)
        recurse(self)
        self._used_names = UsedNamesMapping(dct)
    return self._used_names