import sys
from itertools import product
import numpy as np
import unittest
from numba.core import types
from numba.core.errors import NumbaNotImplementedError
from numba.tests.support import TestCase
from numba.tests.enum_usecases import Shake, RequestError
from numba.np import numpy_support
def test_enum_type(self):

    def check(base_inst, enum_def, type_class):
        np_dt = np.dtype(base_inst)
        nb_ty = numpy_support.from_dtype(np_dt)
        inst = type_class(enum_def, nb_ty)
        recovered = numpy_support.as_dtype(inst)
        self.assertEqual(np_dt, recovered)
    dts = [np.float64, np.int32, np.complex128, np.bool_]
    enums = [Shake, RequestError]
    for dt, enum in product(dts, enums):
        check(dt, enum, types.EnumMember)
    for dt, enum in product(dts, enums):
        check(dt, enum, types.IntEnumMember)