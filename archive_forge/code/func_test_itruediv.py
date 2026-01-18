import pyomo.common.unittest as unittest
from pyomo.contrib.pynumero.dependencies import (
from pyomo.contrib.pynumero.sparse.block_vector import (
def test_itruediv(self):
    v = self.ones
    v /= 3
    self.assertTrue(np.allclose(v.flatten(), np.ones(v.size) / 3))
    v.fill(1.0)
    v /= v
    self.assertTrue(np.allclose(v.flatten(), np.ones(v.size)))
    v.fill(1.0)
    v /= np.ones(v.size) * 2
    self.assertTrue(np.allclose(v.flatten(), np.ones(v.size) / 2))
    v = BlockVector(2)
    a = np.ones(5)
    b = np.arange(9, dtype=np.float64)
    a_copy = a.copy()
    b_copy = b.copy()
    v.set_block(0, a)
    v.set_block(1, b)
    v /= 2.0
    self.assertTrue(np.allclose(v.get_block(0), a_copy / 2.0))
    self.assertTrue(np.allclose(v.get_block(1), b_copy / 2.0))
    v = BlockVector(2)
    a = np.ones(5)
    b = np.zeros(9)
    a_copy = a.copy()
    b_copy = b.copy()
    v.set_block(0, a)
    v.set_block(1, b)
    v2 = BlockVector(2)
    v2.set_block(0, np.ones(5) * 2)
    v2.set_block(1, np.ones(9) * 2)
    v /= v2
    self.assertTrue(np.allclose(v.get_block(0), a_copy / 2))
    self.assertTrue(np.allclose(v.get_block(1), b_copy / 2))
    self.assertTrue(np.allclose(v2.get_block(0), np.ones(5) * 2))
    self.assertTrue(np.allclose(v2.get_block(1), np.ones(9) * 2))
    with self.assertRaises(Exception) as context:
        v *= 'hola'