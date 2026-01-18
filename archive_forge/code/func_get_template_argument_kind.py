from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def get_template_argument_kind(self, num):
    """Returns the TemplateArgumentKind for the indicated template
        argument."""
    return conf.lib.clang_Cursor_getTemplateArgumentKind(self, num)