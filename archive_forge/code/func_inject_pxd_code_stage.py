from __future__ import absolute_import
import itertools
from time import time
from . import Errors
from . import DebugFlags
from . import Options
from .Errors import CompileError, InternalError, AbortError
from . import Naming
def inject_pxd_code_stage(module_node):
    for name, (statlistnode, scope) in context.pxds.items():
        module_node.merge_in(statlistnode, scope, stage='pxd')
    return module_node