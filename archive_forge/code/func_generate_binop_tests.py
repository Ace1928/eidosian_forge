import copy
import itertools
import operator
import unittest
import numpy as np
from numba import jit, njit
from numba.core import types, utils, errors
from numba.core.types.functions import _header_lead
from numba.tests.support import TestCase, tag, needs_blas
from numba.tests.matmul_usecase import (matmul_usecase, imatmul_usecase,
def generate_binop_tests(ns, usecases, tp_runners, npm_array=False):
    for usecase in usecases:
        for tp_name, runner_name in tp_runners.items():
            for nopython in (False, True):
                test_name = 'test_%s_%s' % (usecase, tp_name)
                if nopython:
                    test_name += '_npm'
                flags = Noflags if nopython else force_pyobj_flags
                usecase_name = '%s_usecase' % usecase

                def inner(self, runner_name=runner_name, usecase_name=usecase_name, flags=flags):
                    runner = getattr(self, runner_name)
                    op_usecase = getattr(self.op, usecase_name)
                    runner(op_usecase, flags)
                if nopython and 'array' in tp_name and (not npm_array):

                    def test_meth(self):
                        with self.assertTypingError():
                            inner()
                else:
                    test_meth = inner
                test_meth.__name__ = test_name
                if nopython:
                    test_meth = tag('important')(test_meth)
                ns[test_name] = test_meth