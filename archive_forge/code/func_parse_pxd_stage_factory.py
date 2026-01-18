from __future__ import absolute_import
import itertools
from time import time
from . import Errors
from . import DebugFlags
from . import Options
from .Errors import CompileError, InternalError, AbortError
from . import Naming
def parse_pxd_stage_factory(context, scope, module_name):

    def parse(source_desc):
        tree = context.parse(source_desc, scope, pxd=True, full_module_name=module_name)
        tree.scope = scope
        tree.is_pxd = True
        return tree
    return parse