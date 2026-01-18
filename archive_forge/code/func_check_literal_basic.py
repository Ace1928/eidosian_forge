import numpy as np
import numba
import unittest
from numba.tests.support import TestCase
from numba import njit
from numba.core import types, errors, cgutils
from numba.core.typing import signature
from numba.core.datamodel import models
from numba.core.extending import (
from numba.misc.special import literally
def check_literal_basic(self, literal_args):

    @njit
    def foo(x):
        return literally(x)
    for lit in literal_args:
        self.assertEqual(foo(lit), lit)
    for lit, sig in zip(literal_args, foo.signatures):
        self.assertEqual(sig[0].literal_value, lit)