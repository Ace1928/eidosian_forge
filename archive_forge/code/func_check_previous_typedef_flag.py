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
def check_previous_typedef_flag(self, entry, typedef_flag, pos):
    if typedef_flag != entry.type.typedef_flag:
        error(pos, "'%s' previously declared using '%s'" % (entry.name, ('cdef', 'ctypedef')[entry.type.typedef_flag]))