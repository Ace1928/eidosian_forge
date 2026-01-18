from __future__ import absolute_import
from .Errors import error, message
from . import ExprNodes
from . import Nodes
from . import Builtin
from . import PyrexTypes
from .. import Utils
from .PyrexTypes import py_object_type, unspecified_type
from .Visitor import CythonTransform, EnvTransform
def resolve_assignments(assignments):
    resolved = set()
    for assmt in assignments:
        deps = dependencies[assmt]
        if assmts_resolved.issuperset(deps):
            for node in assmt_to_names[assmt]:
                infer_name_node_type(node)
            inferred_type = assmt.infer_type()
            assmts_resolved.add(assmt)
            resolved.add(assmt)
    assignments.difference_update(resolved)
    return resolved