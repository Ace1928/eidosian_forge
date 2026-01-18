import pyomo.common.unittest as unittest
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as sp
from scipy.sparse import coo_matrix, bmat
from pyomo.contrib.pynumero.sparse import (
import warnings
def test_mul_sparse_matrix(self):
    m = self.basic_m
    flat_prod = m.tocoo() * m.tocoo()
    prod = m * m
    self.assertIsInstance(prod, BlockMatrix)
    self.assertTrue(np.allclose(flat_prod.toarray(), prod.toarray()))
    m2 = m.copy_structure()
    ones = np.ones(m.shape)
    m2.copyfrom(ones)
    flat_prod = m.tocoo() * m2.tocoo()
    prod = m * m2
    self.assertIsInstance(prod, BlockMatrix)
    self.assertTrue(np.allclose(flat_prod.toarray(), prod.toarray()))