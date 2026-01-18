import pyomo.common.unittest as unittest
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as sp
from scipy.sparse import coo_matrix, bmat
from pyomo.contrib.pynumero.sparse import (
import warnings
def test_add_copy(self):
    """
        The purpose of this test is to ensure that copying happens correctly when block matrices are added.
        For example, when adding

        [A  B   +  [D  0
         0  C]      E  F]

        we want to make sure that E and B both get copied in the result rather than just placed in the result.
        """
    bm = self.basic_m.copy()
    bmT = bm.transpose()
    res = bm + bmT
    self.assertIsNot(res.get_block(1, 0), bmT.get_block(1, 0))
    self.assertIsNot(res.get_block(0, 1), bm.get_block(0, 1))
    self.assertTrue(np.allclose(res.toarray(), self.dense + self.dense.transpose()))