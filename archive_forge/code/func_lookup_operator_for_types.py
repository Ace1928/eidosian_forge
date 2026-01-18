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
def lookup_operator_for_types(self, pos, operator, types):
    from .Nodes import Node

    class FakeOperand(Node):
        pass
    operands = [FakeOperand(pos, type=type) for type in types]
    return self.lookup_operator(operator, operands)