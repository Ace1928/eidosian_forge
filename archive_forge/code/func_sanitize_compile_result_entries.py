from collections import namedtuple
import copy
import warnings
from numba.core.tracing import event
from numba.core import (utils, errors, interpreter, bytecode, postproc, config,
from numba.parfors.parfor import ParforDiagnostics
from numba.core.errors import CompilerError
from numba.core.environment import lookup_environment
from numba.core.compiler_machinery import PassManager
from numba.core.untyped_passes import (ExtractByteCode, TranslateByteCode,
from numba.core.typed_passes import (NopythonTypeInference, AnnotateTypes,
from numba.core.object_mode_passes import (ObjectModeFrontEnd,
from numba.core.targetconfig import TargetConfig, Option, ConfigStack
def sanitize_compile_result_entries(entries):
    keys = set(entries.keys())
    fieldset = set(CR_FIELDS)
    badnames = keys - fieldset
    if badnames:
        raise NameError(*badnames)
    missing = fieldset - keys
    for k in missing:
        entries[k] = None
    err = entries['typing_error']
    if err is not None:
        entries['typing_error'] = err.with_traceback(None)
    return entries