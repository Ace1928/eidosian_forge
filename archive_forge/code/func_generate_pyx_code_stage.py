from __future__ import absolute_import
import itertools
from time import time
from . import Errors
from . import DebugFlags
from . import Options
from .Errors import CompileError, InternalError, AbortError
from . import Naming
def generate_pyx_code_stage(module_node):
    module_node.process_implementation(options, result)
    result.compilation_source = module_node.compilation_source
    return result