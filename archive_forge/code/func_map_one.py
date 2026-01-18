from __future__ import absolute_import
import cython
from . import Builtin
from . import ExprNodes
from . import Nodes
from . import Options
from . import PyrexTypes
from .Visitor import TreeVisitor, CythonTransform
from .Errors import error, warning, InternalError
def map_one(self, istate, entry):
    ret = set()
    assmts = self.assmts[entry]
    if istate & assmts.bit:
        if self.is_statically_assigned(entry):
            ret.add(StaticAssignment(entry))
        elif entry.from_closure:
            ret.add(Unknown)
        else:
            ret.add(Uninitialized)
    for assmt in assmts.stats:
        if istate & assmt.bit:
            ret.add(assmt)
    return ret