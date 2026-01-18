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
def side_effect_free_reference(node, setting=False):
    if node.is_name:
        return (node, [])
    elif node.type.is_pyobject and (not setting):
        node = LetRefNode(node)
        return (node, [node])
    elif node.is_subscript:
        base, temps = side_effect_free_reference(node.base)
        index = LetRefNode(node.index)
        return (ExprNodes.IndexNode(node.pos, base=base, index=index), temps + [index])
    elif node.is_attribute:
        obj, temps = side_effect_free_reference(node.obj)
        return (ExprNodes.AttributeNode(node.pos, obj=obj, attribute=node.attribute), temps)
    elif isinstance(node, ExprNodes.BufferIndexNode):
        raise ValueError("Don't allow things like attributes of buffer indexing operations")
    else:
        node = LetRefNode(node)
        return (node, [node])