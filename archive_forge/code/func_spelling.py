from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
@property
def spelling(self):
    """The spelling of this token.

        This is the textual representation of the token in source.
        """
    return conf.lib.clang_getTokenSpelling(self._tu, self)