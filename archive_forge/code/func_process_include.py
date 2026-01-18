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
def process_include(self, inc):
    """
        Add `inc`, which is an instance of `IncludeCode`, to this
        `ModuleScope`. This either adds a new element to the
        `c_includes` dict or it updates an existing entry.

        In detail: the values of the dict `self.c_includes` are
        instances of `IncludeCode` containing the code to be put in the
        generated C file. The keys of the dict are needed to ensure
        uniqueness in two ways: if an include file is specified in
        multiple "cdef extern" blocks, only one `#include` statement is
        generated. Second, the same include might occur multiple times
        if we find it through multiple "cimport" paths. So we use the
        generated code (of the form `#include "header.h"`) as dict key.

        If verbatim code does not belong to any include file (i.e. it
        was put in a `cdef extern from *` block), then we use a unique
        dict key: namely, the `sortkey()`.

        One `IncludeCode` object can contain multiple pieces of C code:
        one optional "main piece" for the include file and several other
        pieces for the verbatim code. The `IncludeCode.dict_update`
        method merges the pieces of two different `IncludeCode` objects
        if needed.
        """
    key = inc.mainpiece()
    if key is None:
        key = inc.sortkey()
    inc.dict_update(self.c_includes, key)
    inc = self.c_includes[key]