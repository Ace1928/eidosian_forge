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
@in_string_annotation
def handleStringAnnotation(self, s, node, ref_lineno, ref_col_offset, err):
    try:
        tree = ast.parse(s)
    except SyntaxError:
        self.report(err, node, s)
        return
    body = tree.body
    if len(body) != 1 or not isinstance(body[0], ast.Expr):
        self.report(err, node, s)
        return
    parsed_annotation = tree.body[0].value
    for descendant in ast.walk(parsed_annotation):
        if 'lineno' in descendant._attributes and 'col_offset' in descendant._attributes:
            descendant.lineno = ref_lineno
            descendant.col_offset = ref_col_offset
    self.handleNode(parsed_annotation, node)