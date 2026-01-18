from __future__ import absolute_import
from .Errors import error, message
from . import ExprNodes
from . import Nodes
from . import Builtin
from . import PyrexTypes
from .. import Utils
from .PyrexTypes import py_object_type, unspecified_type
from .Visitor import CythonTransform, EnvTransform
def resolve_partial(assignments):
    partials = set()
    for assmt in assignments:
        if assmt in partial_assmts:
            continue
        if partial_infer(assmt):
            partials.add(assmt)
            assmts_resolved.add(assmt)
    partial_assmts.update(partials)
    return partials