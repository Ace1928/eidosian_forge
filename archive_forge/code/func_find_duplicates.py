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
def find_duplicates(node):
    if node.is_literal or node.is_name:
        return
    if node in seen_nodes:
        if node not in ref_nodes:
            ref_node = LetRefNode(node)
            ref_nodes[node] = ref_node
            ref_node_sequence.append(ref_node)
    else:
        seen_nodes.add(node)
        if node.is_sequence_constructor:
            for item in node.args:
                find_duplicates(item)