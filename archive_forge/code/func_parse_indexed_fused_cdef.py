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
def parse_indexed_fused_cdef(self, env):
    """
        Interpret fused_cdef_func[specific_type1, ...]

        Note that if this method is called, we are an indexed cdef function
        with fused argument types, and this IndexNode will be replaced by the
        NameNode with specific entry just after analysis of expressions by
        AnalyseExpressionsTransform.
        """
    self.type = PyrexTypes.error_type
    self.is_fused_index = True
    base_type = self.base.type
    positions = []
    if self.index.is_name or self.index.is_attribute:
        positions.append(self.index.pos)
    elif isinstance(self.index, TupleNode):
        for arg in self.index.args:
            positions.append(arg.pos)
    specific_types = self.parse_index_as_types(env, required=False)
    if specific_types is None:
        self.index = self.index.analyse_types(env)
        if not self.base.entry.as_variable:
            error(self.pos, 'Can only index fused functions with types')
        else:
            self.base.entry = self.entry = self.base.entry.as_variable
            self.base.type = self.type = self.entry.type
            self.base.is_temp = True
            self.is_temp = True
            self.entry.used = True
        self.is_fused_index = False
        return
    for i, type in enumerate(specific_types):
        specific_types[i] = type.specialize_fused(env)
    fused_types = base_type.get_fused_types()
    if len(specific_types) > len(fused_types):
        return error(self.pos, 'Too many types specified')
    elif len(specific_types) < len(fused_types):
        t = fused_types[len(specific_types)]
        return error(self.pos, 'Not enough types specified to specialize the function, %s is still fused' % t)
    for pos, specific_type, fused_type in zip(positions, specific_types, fused_types):
        if not any([specific_type.same_as(t) for t in fused_type.types]):
            return error(pos, 'Type not in fused type')
        if specific_type is None or specific_type.is_error:
            return
    fused_to_specific = dict(zip(fused_types, specific_types))
    type = base_type.specialize(fused_to_specific)
    if type.is_fused:
        error(self.pos, 'Index operation makes function only partially specific')
    else:
        for signature in self.base.type.get_all_specialized_function_types():
            if type.same_as(signature):
                self.type = signature
                if self.base.is_attribute:
                    self.entry = signature.entry
                    self.is_attribute = True
                    self.obj = self.base.obj
                self.type.entry.used = True
                self.base.type = signature
                self.base.entry = signature.entry
                break
        else:
            raise InternalError("Couldn't find the right signature")