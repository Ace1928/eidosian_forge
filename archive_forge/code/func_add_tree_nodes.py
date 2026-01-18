import re
import difflib
from collections import namedtuple
import logging
from parso.utils import split_lines
from parso.python.parser import Parser
from parso.python.tree import EndMarker
from parso.python.tokenize import PythonToken, BOM_UTF8_STRING
from parso.python.token import PythonTokenTypes
def add_tree_nodes(self, prefix, children, line_offset=0, last_line_offset_leaf=None):
    if last_line_offset_leaf is None:
        last_line_offset_leaf = children[-1].get_last_leaf()
    group = self._ChildrenGroup(prefix, children, line_offset, last_line_offset_leaf)
    self._children_groups.append(group)