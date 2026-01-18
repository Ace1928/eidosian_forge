import os, sys, subprocess
import dis
import itertools
import numpy as np
import numba
from numba import jit, njit
from numba.core import errors, ir, types, typing, typeinfer, utils
from numba.core.typeconv import Conversion
from numba.extending import overload_method
from numba.tests.support import TestCase, tag
from numba.tests.test_typeconv import CompatibilityTestMixin
from numba.core.untyped_passes import TranslateByteCode, IRProcessing
from numba.core.typed_passes import PartialTypeInference
from numba.core.compiler_machinery import FunctionPass, register_pass
import unittest
def test_issue_6293(self):
    """https://github.com/numba/numba/issues/6293

        Typer does not propagate return type to all return variables
        """

    @jit(nopython=True)
    def confuse_typer(x):
        if x == x:
            return int(x)
        else:
            return x
    confuse_typer.compile((types.float64,))
    cres = confuse_typer.overloads[types.float64,]
    typemap = cres.type_annotation.typemap
    return_vars = {}
    for block in cres.type_annotation.blocks.values():
        for inst in block.body:
            if isinstance(inst, ir.Return):
                varname = inst.value.name
                return_vars[varname] = typemap[varname]
    self.assertTrue(all((vt == types.float64 for vt in return_vars.values())))