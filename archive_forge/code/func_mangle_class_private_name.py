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
def mangle_class_private_name(self, name):
    if name and name.lower().startswith('__pyx_'):
        return name
    if name and name.startswith('__') and (not name.endswith('__')):
        name = EncodedString('_%s%s' % (self.class_name.lstrip('_'), name))
    return name