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
def check_c_classes(self):
    debug_check_c_classes = 0
    if debug_check_c_classes:
        print('Scope.check_c_classes: checking scope ' + self.qualified_name)
    for entry in self.c_class_entries:
        if debug_check_c_classes:
            print('...entry %s %s' % (entry.name, entry))
            print('......type = ', entry.type)
            print('......visibility = ', entry.visibility)
        self.check_c_class(entry)