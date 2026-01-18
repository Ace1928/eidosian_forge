from __future__ import absolute_import
import cython
from . import Builtin
from . import ExprNodes
from . import Nodes
from . import Options
from . import PyrexTypes
from .Visitor import TreeVisitor, CythonTransform
from .Errors import error, warning, InternalError
def newblock(self, parent=None):
    """Create floating block linked to `parent` if given.

           NOTE: Block is NOT added to self.blocks
        """
    block = ControlBlock()
    self.blocks.add(block)
    if parent:
        parent.add_child(block)
    return block