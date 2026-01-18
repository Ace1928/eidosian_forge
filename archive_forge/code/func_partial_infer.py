from __future__ import absolute_import
from .Errors import error, message
from . import ExprNodes
from . import Nodes
from . import Builtin
from . import PyrexTypes
from .. import Utils
from .PyrexTypes import py_object_type, unspecified_type
from .Visitor import CythonTransform, EnvTransform
def partial_infer(assmt):
    partial_types = []
    for node in assmt_to_names[assmt]:
        partial_type = infer_name_node_type_partial(node)
        if partial_type is None:
            return False
        partial_types.append((node, partial_type))
    for node, partial_type in partial_types:
        node.inferred_type = partial_type
    assmt.infer_type()
    return True