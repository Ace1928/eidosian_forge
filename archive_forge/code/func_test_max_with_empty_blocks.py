import pyomo.common.unittest as unittest
from pyomo.contrib.pynumero.dependencies import (
from pyomo.contrib.pynumero.sparse.block_vector import (
def test_max_with_empty_blocks(self):
    b = BlockVector(3)
    b.set_block(0, np.zeros(3))
    b.set_block(1, np.zeros(0))
    b.set_block(2, np.zeros(3))
    self.assertEqual(b.max(), 0)