from __future__ import absolute_import
import cython
import re
import sys
import copy
import os.path
import operator
from .Errors import (
from .Code import UtilityCode, TempitaUtilityCode
from . import StringEncoding
from . import Naming
from . import Nodes
from .Nodes import Node, utility_code_for_imports, SingleAssignmentNode
from . import PyrexTypes
from .PyrexTypes import py_object_type, typecast, error_type, \
from . import TypeSlots
from .Builtin import (
from . import Builtin
from . import Symtab
from .. import Utils
from .Annotate import AnnotationItem
from . import Future
from ..Debugging import print_call_chain
from .DebugFlags import debug_disposal_code, debug_coercion
from .Pythran import (to_pythran, is_pythran_supported_type, is_pythran_supported_operation_type,
from .PyrexTypes import PythranExpr
@classmethod
def for_cproperty(cls, pos, obj, entry):
    property_scope = entry.scope
    getter_entry = property_scope.lookup_here(entry.name)
    assert getter_entry, 'Getter not found in scope %s: %s' % (property_scope, property_scope.entries)
    function = NameNode(pos, name=entry.name, entry=getter_entry, type=getter_entry.type)
    node = cls(pos, function=function, args=[obj])
    return node