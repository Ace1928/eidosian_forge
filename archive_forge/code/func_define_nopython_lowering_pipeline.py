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
@staticmethod
def define_nopython_lowering_pipeline(state, name='nopython_lowering'):
    pm = PassManager(name)
    pm.add_pass(NoPythonSupportedFeatureValidation, 'ensure features that are in use are in a valid form')
    pm.add_pass(IRLegalization, 'ensure IR is legal prior to lowering')
    pm.add_pass(AnnotateTypes, 'annotate types')
    if state.flags.auto_parallel.enabled:
        pm.add_pass(NativeParforLowering, 'native parfor lowering')
    else:
        pm.add_pass(NativeLowering, 'native lowering')
    pm.add_pass(NoPythonBackend, 'nopython mode backend')
    pm.add_pass(DumpParforDiagnostics, 'dump parfor diagnostics')
    pm.finalize()
    return pm