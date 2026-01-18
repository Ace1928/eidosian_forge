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
def declare_type(self, name, type, pos, cname=None, visibility='private', api=0, defining=1, shadow=0, template=0):
    if not cname:
        cname = name
    entry = self.declare(name, cname, type, pos, visibility, shadow, is_type=True)
    entry.is_type = 1
    entry.api = api
    if defining:
        self.type_entries.append(entry)
    if not template and getattr(type, 'entry', None) is None:
        type.entry = entry
    return entry