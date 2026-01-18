from symengine import symbols, init_printing
from symengine.lib.symengine_wrapper import (DenseMatrix, Symbol, Integer,
from symengine.test_utilities import raises
import unittest
@unittest.skipIf(not have_numpy, 'requires numpy')
def test_dump_complex():
    ref = [1j, 2j, 3j, 4j]
    A = DenseMatrix(2, 2, ref)
    out = np.empty(4, dtype=np.complex128)
    A.dump_complex(out)
    assert np.allclose(out, ref)