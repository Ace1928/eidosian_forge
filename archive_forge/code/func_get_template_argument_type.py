from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def get_template_argument_type(self, num):
    return conf.lib.clang_Type_getTemplateArgumentAsType(self, num)