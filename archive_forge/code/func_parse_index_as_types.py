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
def parse_index_as_types(self, env, required=True):
    if isinstance(self.index, TupleNode):
        indices = self.index.args
    else:
        indices = [self.index]
    type_indices = []
    for index in indices:
        type_indices.append(index.analyse_as_type(env))
        if type_indices[-1] is None:
            if required:
                error(index.pos, 'not parsable as a type')
            return None
    return type_indices