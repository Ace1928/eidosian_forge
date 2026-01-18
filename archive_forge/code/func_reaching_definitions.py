from __future__ import absolute_import
import cython
from . import Builtin
from . import ExprNodes
from . import Nodes
from . import Options
from . import PyrexTypes
from .Visitor import TreeVisitor, CythonTransform
from .Errors import error, warning, InternalError
def reaching_definitions(self):
    """Per-block reaching definitions analysis."""
    dirty = True
    while dirty:
        dirty = False
        for block in self.blocks:
            i_input = 0
            for parent in block.parents:
                i_input |= parent.i_output
            i_output = i_input & ~block.i_kill | block.i_gen
            if i_output != block.i_output:
                dirty = True
            block.i_input = i_input
            block.i_output = i_output