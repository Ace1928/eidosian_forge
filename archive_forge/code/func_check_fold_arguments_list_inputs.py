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
def check_fold_arguments_list_inputs(self, func, args, kws):

    def make_tuple(*args):
        return args
    unused_handler = None
    pysig = utils.pysignature(func)
    names = list(pysig.parameters)
    with self.subTest(kind='dict'):
        folded_dict = typing.fold_arguments(pysig, args, kws, make_tuple, unused_handler, unused_handler)
        for i, (j, k) in enumerate(zip(folded_dict, names)):
            got_index, got_param, got_name = j
            self.assertEqual(got_index, i)
            self.assertEqual(got_name, f'arg.{k}')
    kws = list(kws.items())
    with self.subTest(kind='list'):
        folded_list = typing.fold_arguments(pysig, args, kws, make_tuple, unused_handler, unused_handler)
        self.assertEqual(folded_list, folded_dict)