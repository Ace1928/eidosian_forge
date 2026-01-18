import numpy as np
import sys
import traceback
from numba import jit, njit
from numba.core import types, errors
from numba.tests.support import (TestCase, expected_failure_py311,
import unittest
def check_reraise(self, flags):

    def raise_exc(exc):
        raise exc
    pyfunc = reraise
    cfunc = jit((), **flags)(pyfunc)
    for op, err in [(lambda: raise_exc(ZeroDivisionError), ZeroDivisionError), (lambda: raise_exc(UDEArgsToSuper('msg', 1)), UDEArgsToSuper), (lambda: raise_exc(UDENoArgSuper('msg', 1)), UDENoArgSuper)]:

        def gen_impl(fn):

            def impl():
                try:
                    op()
                except err:
                    fn()
            return impl
        pybased = gen_impl(pyfunc)
        cbased = gen_impl(cfunc)
        self.check_against_python(flags, pybased, cbased, err)