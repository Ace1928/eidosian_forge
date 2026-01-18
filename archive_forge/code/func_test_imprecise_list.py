import math
import re
import textwrap
import operator
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.core.errors import TypingError
from numba.core.types.functions import _header_lead
from numba.tests.support import TestCase
def test_imprecise_list(self):
    """
        Type inference should catch that a list type's remain imprecise,
        instead of letting lowering fail.
        """
    with self.assertRaises(TypingError) as raises:
        njit(())(imprecise_list)
    errmsg = str(raises.exception)
    msg = "Cannot infer the type of variable 'l', have imprecise type: list(undefined)"
    self.assertIn(msg, errmsg)
    self.assertIn('For Numba to be able to compile a list', errmsg)