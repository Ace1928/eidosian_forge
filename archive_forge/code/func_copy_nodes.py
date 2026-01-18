import re
import difflib
from collections import namedtuple
import logging
from parso.utils import split_lines
from parso.python.parser import Parser
from parso.python.tree import EndMarker
from parso.python.tokenize import PythonToken, BOM_UTF8_STRING
from parso.python.token import PythonTokenTypes
def copy_nodes(self, tree_nodes, until_line, line_offset):
    """
        Copies tree nodes from the old parser tree.

        Returns the number of tree nodes that were copied.
        """
    if tree_nodes[0].type in ('error_leaf', 'error_node'):
        return []
    indentation = _get_indentation(tree_nodes[0])
    old_working_stack = list(self._working_stack)
    old_prefix = self.prefix
    old_indents = self.indents
    self.indents = [i for i in self.indents if i <= indentation]
    self._update_insertion_node(indentation)
    new_nodes, self._working_stack, self.prefix, added_indents = self._copy_nodes(list(self._working_stack), tree_nodes, until_line, line_offset, self.prefix)
    if new_nodes:
        self.indents += added_indents
    else:
        self._working_stack = old_working_stack
        self.prefix = old_prefix
        self.indents = old_indents
    return new_nodes