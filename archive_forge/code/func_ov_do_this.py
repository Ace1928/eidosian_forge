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
@overload(do_this)
def ov_do_this(x, y):
    if isinstance(y, types.IntegerLiteral):
        raise errors.NumbaValueError('oops')
    else:

        def impl(x, y):
            return hidden(x, y)
        return impl