from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
@property
def exception_specification_kind(self):
    """
        Retrieve the exception specification kind, which is one of the values
        from the ExceptionSpecificationKind enumeration.
        """
    if not hasattr(self, '_exception_specification_kind'):
        exc_kind = conf.lib.clang_getCursorExceptionSpecificationType(self)
        self._exception_specification_kind = ExceptionSpecificationKind.from_id(exc_kind)
    return self._exception_specification_kind