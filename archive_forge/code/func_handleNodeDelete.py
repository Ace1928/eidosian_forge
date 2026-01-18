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
def handleNodeDelete(self, node):

    def on_conditional_branch():
        """
            Return `True` if node is part of a conditional body.
            """
        current = getattr(node, '_pyflakes_parent', None)
        while current:
            if isinstance(current, (ast.If, ast.While, ast.IfExp)):
                return True
            current = getattr(current, '_pyflakes_parent', None)
        return False
    name = getNodeName(node)
    if not name:
        return
    if on_conditional_branch():
        return
    if isinstance(self.scope, FunctionScope) and name in self.scope.globals:
        self.scope.globals.remove(name)
    else:
        try:
            del self.scope[name]
        except KeyError:
            self.report(messages.UndefinedName, node, name)