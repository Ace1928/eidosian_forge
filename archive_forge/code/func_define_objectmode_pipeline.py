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
def define_objectmode_pipeline(state, name='object'):
    """Returns an object-mode pipeline based PassManager
        """
    pm = PassManager(name)
    if state.func_ir is None:
        pm.add_pass(TranslateByteCode, 'analyzing bytecode')
        pm.add_pass(FixupArgs, 'fix up args')
    else:
        pm.add_pass(PreLowerStripPhis, 'remove phis nodes')
    pm.add_pass(IRProcessing, 'processing IR')
    pm.add_pass(CanonicalizeLoopEntry, 'canonicalize loop entry')
    pm.add_pass(CanonicalizeLoopExit, 'canonicalize loop exit')
    pm.add_pass(ObjectModeFrontEnd, 'object mode frontend')
    pm.add_pass(InlineClosureLikes, 'inline calls to locally defined closures')
    pm.add_pass(MakeFunctionToJitFunction, 'convert make_function into JIT functions')
    pm.add_pass(IRLegalization, 'ensure IR is legal prior to lowering')
    pm.add_pass(AnnotateTypes, 'annotate types')
    pm.add_pass(ObjectModeBackEnd, 'object mode backend')
    pm.finalize()
    return pm