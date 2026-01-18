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
def declare_builtin_cfunction(self, name, type, cname, utility_code=None):
    name = EncodedString(name)
    entry = self.declare_cfunction(name, type, pos=None, cname=cname, visibility='extern', utility_code=utility_code)
    var_entry = Entry(name, name, py_object_type)
    var_entry.qualified_name = name
    var_entry.is_variable = 1
    var_entry.is_builtin = 1
    var_entry.utility_code = utility_code
    var_entry.scope = entry.scope
    entry.as_variable = var_entry
    return entry