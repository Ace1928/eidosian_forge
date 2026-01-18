from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def get_exception_specification_kind(self):
    """
        Return the kind of the exception specification; a value from
        the ExceptionSpecificationKind enumeration.
        """
    return ExceptionSpecificationKind.from_id(conf.lib.clang.getExceptionSpecificationType(self))