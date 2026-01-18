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
def getScopeNode(self, node):
    return self._getAncestor(node, tuple(Checker._ast_node_scope.keys()))