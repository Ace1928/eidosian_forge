import pyomo.common.unittest as unittest
from pyomo.contrib.pynumero.dependencies import (
def test_nblocks(self):
    v1 = self.v1
    self.assertEqual(v1.nblocks, 4)
    v2 = self.v2
    self.assertEqual(v2.nblocks, 7)