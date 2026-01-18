from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
@property
def displayname(self):
    """
        Return the display name for the entity referenced by this cursor.

        The display name contains extra information that helps identify the
        cursor, such as the parameters of a function or template or the
        arguments of a class template specialization.
        """
    if not hasattr(self, '_displayname'):
        self._displayname = conf.lib.clang_getCursorDisplayName(self)
    return self._displayname