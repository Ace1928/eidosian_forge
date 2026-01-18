from __future__ import absolute_import
import cython
import copy
import hashlib
import sys
from . import PyrexTypes
from . import Naming
from . import ExprNodes
from . import Nodes
from . import Options
from . import Builtin
from . import Errors
from .Visitor import VisitorTransform, TreeVisitor
from .Visitor import CythonTransform, EnvTransform, ScopeTrackingTransform
from .UtilNodes import LetNode, LetRefNode
from .TreeFragment import TreeFragment
from .StringEncoding import EncodedString, _unicode
from .Errors import error, warning, CompileError, InternalError
from .Code import UtilityCode
@staticmethod
def chain_decorators(node, decorators, name):
    """
        Decorators are applied directly in DefNode and PyClassDefNode to avoid
        reassignments to the function/class name - except for cdef class methods.
        For those, the reassignment is required as methods are originally
        defined in the PyMethodDef struct.

        The IndirectionNode allows DefNode to override the decorator.
        """
    decorator_result = ExprNodes.NameNode(node.pos, name=name)
    for decorator in decorators[::-1]:
        decorator_result = ExprNodes.SimpleCallNode(decorator.pos, function=decorator.decorator, args=[decorator_result])
    name_node = ExprNodes.NameNode(node.pos, name=name)
    reassignment = Nodes.SingleAssignmentNode(node.pos, lhs=name_node, rhs=decorator_result)
    reassignment = Nodes.IndirectionNode([reassignment])
    node.decorator_indirection = reassignment
    return [node, reassignment]