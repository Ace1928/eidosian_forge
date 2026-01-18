from __future__ import absolute_import
import cython
from . import Builtin
from . import ExprNodes
from . import Nodes
from . import Options
from . import PyrexTypes
from .Visitor import TreeVisitor, CythonTransform
from .Errors import error, warning, InternalError
def mark_position(self, node):
    """Mark position if DOT output is enabled."""
    if self.current_directives['control_flow.dot_output']:
        self.flow.mark_position(node)