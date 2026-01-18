import numba
import numba.parfors.parfor
from numba import njit
from numba.core import ir_utils
from numba.core import types, ir,  compiler
from numba.core.registry import cpu_target
from numba.core.ir_utils import (copy_propagate, apply_copy_propagate,
from numba.core.typed_passes import type_inference_stage
from numba.core.compiler_machinery import FunctionPass, register_pass, PassManager
from numba.core.untyped_passes import (ExtractByteCode, TranslateByteCode, FixupArgs,
from numba.core.typed_passes import (NopythonTypeInference, AnnotateTypes,
import numpy as np
from numba.tests.support import skip_parfors_unsupported, needs_blas
import unittest
def seed_call_exists(func_ir):
    for inst in func_ir.blocks[0].body:
        if isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Expr) and (inst.value.op == 'call') and (func_ir.get_definition(inst.value.func).attr == 'seed'):
            return True
    return False