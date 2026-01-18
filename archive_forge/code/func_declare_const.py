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
def declare_const(self, name, type, value, pos, cname=None, visibility='private', api=0, create_wrapper=0):
    if not cname:
        if self.in_cinclude or (visibility == 'public' or api):
            cname = name
        else:
            cname = self.mangle(Naming.enum_prefix, name)
    entry = self.declare(name, cname, type, pos, visibility, create_wrapper=create_wrapper)
    entry.is_const = 1
    entry.value_node = value
    return entry