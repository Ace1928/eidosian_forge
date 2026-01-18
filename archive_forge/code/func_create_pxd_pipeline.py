from __future__ import absolute_import
import itertools
from time import time
from . import Errors
from . import DebugFlags
from . import Options
from .Errors import CompileError, InternalError, AbortError
from . import Naming
def create_pxd_pipeline(context, scope, module_name):
    from .CodeGeneration import ExtractPxdCode
    return [parse_pxd_stage_factory(context, scope, module_name)] + create_pipeline(context, 'pxd') + [ExtractPxdCode()]