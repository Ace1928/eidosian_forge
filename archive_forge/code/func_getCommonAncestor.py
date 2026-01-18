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
def getCommonAncestor(self, lnode, rnode, stop):
    if stop in (lnode, rnode) or not (hasattr(lnode, '_pyflakes_parent') and hasattr(rnode, '_pyflakes_parent')):
        return None
    if lnode is rnode:
        return lnode
    if lnode._pyflakes_depth > rnode._pyflakes_depth:
        return self.getCommonAncestor(lnode._pyflakes_parent, rnode, stop)
    if lnode._pyflakes_depth < rnode._pyflakes_depth:
        return self.getCommonAncestor(lnode, rnode._pyflakes_parent, stop)
    return self.getCommonAncestor(lnode._pyflakes_parent, rnode._pyflakes_parent, stop)