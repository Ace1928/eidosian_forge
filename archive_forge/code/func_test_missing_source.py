import sys
import subprocess
import numpy as np
import os
import warnings
from numba import jit, njit, types
from numba.core import errors
from numba.experimental import structref
from numba.extending import (overload, intrinsic, overload_method,
from numba.core.compiler import CompilerBase
from numba.core.untyped_passes import (TranslateByteCode, FixupArgs,
from numba.core.typed_passes import (NopythonTypeInference, DeadCodeElimination,
from numba.core.compiler_machinery import PassManager
from numba.core.types.functions import _err_reasons as error_reasons
from numba.tests.support import (skip_parfors_unsupported, override_config,
import unittest
def test_missing_source(self):

    @structref.register
    class ParticleType(types.StructRef):
        pass

    class Particle(structref.StructRefProxy):

        def __new__(cls, pos, mass):
            return structref.StructRefProxy.__new__(cls, pos)
    structref.define_proxy(Particle, ParticleType, ['pos', 'mass'])
    with self.assertRaises(errors.TypingError) as raises:
        Particle(pos=1, mass=2)
    excstr = str(raises.exception)
    self.assertIn("missing a required argument: 'mass'", excstr)