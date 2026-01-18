from __future__ import absolute_import
from .Errors import error, message
from . import ExprNodes
from . import Nodes
from . import Builtin
from . import PyrexTypes
from .. import Utils
from .PyrexTypes import py_object_type, unspecified_type
from .Visitor import CythonTransform, EnvTransform
def infer_name_node_type(node):
    types = [assmt.inferred_type for assmt in node.cf_state]
    if not types:
        node_type = py_object_type
    else:
        entry = node.entry
        node_type = spanning_type(types, entry.might_overflow, scope)
    node.inferred_type = node_type