import pyomo.common.unittest as unittest
from pyomo.contrib.pynumero.dependencies import (
from pyomo.contrib.pynumero.sparse.block_vector import (
def test_has_none(self):
    v = self.ones
    self.assertFalse(v.has_none)
    v = BlockVector(3)
    v.set_block(0, np.ones(2))
    v.set_block(2, np.ones(3))
    self.assertTrue(v.has_none)
    v.set_block(1, np.ones(2))
    self.assertFalse(v.has_none)