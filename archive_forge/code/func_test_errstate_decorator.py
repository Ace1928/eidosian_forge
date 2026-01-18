import pytest
import sysconfig
import numpy as np
from numpy.testing import assert_, assert_raises, IS_WASM
def test_errstate_decorator(self):

    @np.errstate(all='ignore')
    def foo():
        a = -np.arange(3)
        a // 0
    foo()