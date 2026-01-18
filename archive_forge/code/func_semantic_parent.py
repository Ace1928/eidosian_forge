from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
@property
def semantic_parent(self):
    """Return the semantic parent for this cursor."""
    if not hasattr(self, '_semantic_parent'):
        self._semantic_parent = conf.lib.clang_getCursorSemanticParent(self)
    return self._semantic_parent