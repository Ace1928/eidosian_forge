from __future__ import absolute_import
import re
import copy
import operator
from ..Utils import try_finally_contextmanager
from .Errors import warning, error, InternalError, performance_hint
from .StringEncoding import EncodedString
from . import Options, Naming
from . import PyrexTypes
from .PyrexTypes import py_object_type, unspecified_type
from .TypeSlots import (
from . import Future
from . import Code
def add_imported_module(self, scope):
    if scope not in self.cimported_modules:
        for inc in scope.c_includes.values():
            self.process_include(inc)
        self.cimported_modules.append(scope)
        for m in scope.cimported_modules:
            self.add_imported_module(m)