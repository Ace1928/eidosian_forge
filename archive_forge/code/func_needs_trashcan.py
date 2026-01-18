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
def needs_trashcan(self):
    directive = self.directives.get('trashcan')
    if directive is False:
        return False
    if directive and self.has_cyclic_pyobject_attrs:
        return True
    base_type = self.parent_type.base_type
    if base_type and base_type.scope is not None:
        return base_type.scope.needs_trashcan()
    return self.parent_type.builtin_trashcan