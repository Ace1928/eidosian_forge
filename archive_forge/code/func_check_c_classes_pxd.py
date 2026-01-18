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
def check_c_classes_pxd(self):
    for entry in self.c_class_entries:
        if not entry.type.scope:
            error(entry.pos, "C class '%s' is declared but not defined" % entry.name)