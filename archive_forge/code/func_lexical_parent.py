from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
@property
def lexical_parent(self):
    """Return the lexical parent for this cursor."""
    if not hasattr(self, '_lexical_parent'):
        self._lexical_parent = conf.lib.clang_getCursorLexicalParent(self)
    return self._lexical_parent