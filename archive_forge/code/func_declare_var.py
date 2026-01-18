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
def declare_var(self, name, type, pos, cname=None, visibility='extern', pytyping_modifiers=None):
    if not cname:
        cname = name
    self._reject_pytyping_modifiers(pos, pytyping_modifiers)
    entry = self.declare(name, cname, type, pos, visibility)
    entry.is_variable = True
    return entry