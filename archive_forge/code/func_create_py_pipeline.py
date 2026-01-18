from __future__ import absolute_import
import itertools
from time import time
from . import Errors
from . import DebugFlags
from . import Options
from .Errors import CompileError, InternalError, AbortError
from . import Naming
def create_py_pipeline(context, options, result):
    return create_pyx_pipeline(context, options, result, py=True)