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
def analyse_as_buffer_operation(self, env, getting):
    """
        Analyse buffer indexing and memoryview indexing/slicing
        """
    if isinstance(self.index, TupleNode):
        indices = self.index.args
    else:
        indices = [self.index]
    base = self.base
    base_type = base.type
    replacement_node = None
    if base_type.is_memoryviewslice:
        from . import MemoryView
        if base.is_memview_slice:
            merged_indices = base.merged_indices(indices)
            if merged_indices is not None:
                base = base.base
                base_type = base.type
                indices = merged_indices
        have_slices, indices, newaxes = MemoryView.unellipsify(indices, base_type.ndim)
        if have_slices:
            replacement_node = MemoryViewSliceNode(self.pos, indices=indices, base=base)
        else:
            replacement_node = MemoryViewIndexNode(self.pos, indices=indices, base=base)
    elif base_type.is_buffer or base_type.is_pythran_expr:
        if base_type.is_pythran_expr or len(indices) == base_type.ndim:
            is_buffer_access = True
            indices = [index.analyse_types(env) for index in indices]
            if base_type.is_pythran_expr:
                do_replacement = all((index.type.is_int or index.is_slice or index.type.is_pythran_expr for index in indices))
                if do_replacement:
                    for i, index in enumerate(indices):
                        if index.is_slice:
                            index = SliceIntNode(index.pos, start=index.start, stop=index.stop, step=index.step)
                            index = index.analyse_types(env)
                            indices[i] = index
            else:
                do_replacement = all((index.type.is_int for index in indices))
            if do_replacement:
                replacement_node = BufferIndexNode(self.pos, indices=indices, base=base)
                assert not isinstance(self.index, CloneNode)
    if replacement_node is not None:
        replacement_node = replacement_node.analyse_types(env, getting)
    return replacement_node