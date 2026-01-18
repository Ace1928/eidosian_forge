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
def declare_cgetter(self, name, return_type, pos=None, cname=None, visibility='private', modifiers=(), defining=False, **cfunc_type_config):
    assert all((k in ('exception_value', 'exception_check', 'nogil', 'with_gil', 'is_const_method', 'is_static_method') for k in cfunc_type_config))
    cfunc_type = PyrexTypes.CFuncType(return_type, [PyrexTypes.CFuncTypeArg('self', self.parent_type, None)], **cfunc_type_config)
    entry = self.declare_cfunction(name, cfunc_type, pos, cname=None, visibility=visibility, modifiers=modifiers, defining=defining)
    entry.is_cgetter = True
    if cname is not None:
        entry.func_cname = cname
    return entry