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
def declare_typedef(self, name, base_type, pos, cname=None, visibility='private', api=0):
    if not cname:
        if self.in_cinclude or (visibility != 'private' or api):
            cname = name
        else:
            cname = self.mangle(Naming.type_prefix, name)
    try:
        if self.is_cpp_class_scope:
            namespace = self.outer_scope.lookup(self.name).type
        else:
            namespace = None
        type = PyrexTypes.create_typedef_type(name, base_type, cname, visibility == 'extern', namespace)
    except ValueError as e:
        error(pos, e.args[0])
        type = PyrexTypes.error_type
    entry = self.declare_type(name, type, pos, cname, visibility=visibility, api=api)
    type.qualified_name = entry.qualified_name
    return entry