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
def maybe_ref(arg):
    if arg.type.is_cpp_class and (not arg.type.is_reference):
        return PyrexTypes.CFuncTypeArg(arg.name, PyrexTypes.c_ref_type(arg.type), arg.pos)
    else:
        return arg