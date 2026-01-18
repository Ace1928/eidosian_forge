import pyomo.common.unittest as unittest
from pyomo.contrib.pynumero.dependencies import (
from pyomo.contrib.pynumero.sparse.block_vector import (
def test_bshape(self):
    self.assertEqual(self.ones.bshape, (3,))