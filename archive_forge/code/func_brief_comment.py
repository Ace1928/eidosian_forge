from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
@property
def brief_comment(self):
    """Returns the brief comment text associated with that Cursor"""
    return conf.lib.clang_Cursor_getBriefCommentText(self)