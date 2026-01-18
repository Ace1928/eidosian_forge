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
@try_finally_contextmanager
def new_c_type_context(self, in_c_type_context=None):
    old_c_type_context = self.in_c_type_context
    if in_c_type_context is not None:
        self.in_c_type_context = in_c_type_context
    yield
    self.in_c_type_context = old_c_type_context