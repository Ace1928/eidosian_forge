import re
import numpy as np
from numba.tests.support import (TestCase, override_config, captured_stdout,
from numba import jit, njit
from numba.core import types, ir, postproc, compiler
from numba.core.ir_utils import (guard, find_callname, find_const,
from numba.core.registry import CPUDispatcher
from numba.core.inline_closurecall import inline_closure_call
from numba.core.untyped_passes import (ExtractByteCode, TranslateByteCode, FixupArgs,
from numba.core.typed_passes import (NopythonTypeInference, AnnotateTypes,
from numba.core.compiler_machinery import FunctionPass, PassManager, register_pass
import unittest
def gen_pipeline(state, test_pass):
    name = 'inline_test'
    pm = PassManager(name)
    pm.add_pass(TranslateByteCode, 'analyzing bytecode')
    pm.add_pass(FixupArgs, 'fix up args')
    pm.add_pass(IRProcessing, 'processing IR')
    pm.add_pass(WithLifting, 'Handle with contexts')
    if not state.flags.no_rewrites:
        pm.add_pass(GenericRewrites, 'nopython rewrites')
        pm.add_pass(RewriteSemanticConstants, 'rewrite semantic constants')
        pm.add_pass(DeadBranchPrune, 'dead branch pruning')
    pm.add_pass(InlineClosureLikes, 'inline calls to locally defined closures')
    pm.add_pass(NopythonTypeInference, 'nopython frontend')
    if state.flags.auto_parallel.enabled:
        pm.add_pass(PreParforPass, 'Preprocessing for parfors')
    if not state.flags.no_rewrites:
        pm.add_pass(NopythonRewrites, 'nopython rewrites')
    if state.flags.auto_parallel.enabled:
        pm.add_pass(ParforPass, 'convert to parfors')
        pm.add_pass(ParforFusionPass, 'fuse parfors')
        pm.add_pass(ParforPreLoweringPass, 'parfor prelowering')
    pm.add_pass(test_pass, 'inline test')
    pm.add_pass(IRLegalization, 'ensure IR is legal prior to lowering')
    pm.add_pass(AnnotateTypes, 'annotate types')
    pm.add_pass(PreserveIR, 'preserve IR')
    if state.flags.auto_parallel.enabled:
        pm.add_pass(NativeParforLowering, 'native parfor lowering')
    else:
        pm.add_pass(NativeLowering, 'native lowering')
    pm.add_pass(NoPythonBackend, 'nopython mode backend')
    pm.add_pass(DumpParforDiagnostics, 'dump parfor diagnostics')
    return pm