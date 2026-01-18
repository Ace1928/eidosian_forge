from symengine import symbols, init_printing
from symengine.lib.symengine_wrapper import (DenseMatrix, Symbol, Integer,
from symengine.test_utilities import raises
import unittest
@unittest.skipIf(not have_numpy, 'requires numpy')
def test_dump_real():
    ref = [1, 2, 3, 4]
    A = DenseMatrix(2, 2, ref)
    out = np.empty(4)
    A.dump_real(out)
    assert np.allclose(out, ref)