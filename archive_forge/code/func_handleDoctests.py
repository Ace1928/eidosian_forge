import __future__
import builtins
import ast
import collections
import contextlib
import doctest
import functools
import os
import re
import string
import sys
import warnings
from pyflakes import messages
def handleDoctests(self, node):
    try:
        docstring, node_lineno = self.getDocstring(node.body[0])
        examples = docstring and self._getDoctestExamples(docstring)
    except (ValueError, IndexError):
        return
    if not examples:
        return
    saved_stack = self.scopeStack
    self.scopeStack = [self.scopeStack[0]]
    node_offset = self.offset or (0, 0)
    with self.in_scope(DoctestScope):
        if '_' not in self.scopeStack[0]:
            self.addBinding(None, Builtin('_'))
        for example in examples:
            try:
                tree = ast.parse(example.source, '<doctest>')
            except SyntaxError as e:
                position = (node_lineno + example.lineno + e.lineno, example.indent + 4 + (e.offset or 0))
                self.report(messages.DoctestSyntaxError, node, position)
            else:
                self.offset = (node_offset[0] + node_lineno + example.lineno, node_offset[1] + example.indent + 4)
                self.handleChildren(tree)
                self.offset = node_offset
    self.scopeStack = saved_stack