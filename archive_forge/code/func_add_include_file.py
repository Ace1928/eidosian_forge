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
def add_include_file(self, filename, verbatim_include=None, late=False):
    """
        Add `filename` as include file. Add `verbatim_include` as
        verbatim text in the C file.
        Both `filename` and `verbatim_include` can be `None` or empty.
        """
    inc = Code.IncludeCode(filename, verbatim_include, late=late)
    self.process_include(inc)