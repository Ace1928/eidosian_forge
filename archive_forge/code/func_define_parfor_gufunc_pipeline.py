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
def define_parfor_gufunc_pipeline(state, name='parfor_gufunc_typed'):
    """Returns the typed part of the nopython pipeline"""
    pm = PassManager(name)
    assert state.func_ir
    pm.add_pass(IRProcessing, 'processing IR')
    pm.add_pass(NopythonTypeInference, 'nopython frontend')
    pm.add_pass(ParforPreLoweringPass, 'parfor prelowering')
    pm.finalize()
    return pm