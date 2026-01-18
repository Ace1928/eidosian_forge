from __future__ import absolute_import
import cython
from . import Builtin
from . import ExprNodes
from . import Nodes
from . import Options
from . import PyrexTypes
from .Visitor import TreeVisitor, CythonTransform
from .Errors import error, warning, InternalError
def find_in_stack(self, env):
    if env == self.env:
        return self.flow
    for e, flow in reversed(self.stack):
        if e is env:
            return flow
    assert False