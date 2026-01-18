import pyomo.common.unittest as unittest
from pyomo.contrib.pynumero.dependencies import (
def test_min_max_with_size0_blocks(self):
    v = MPIBlockVector(3, [0, 1, 2], comm)
    rank = comm.Get_rank()
    if rank == 0:
        v.set_block(0, np.array([8, 4, 7, 12]))
    if rank == 1:
        v.set_block(1, np.array([]))
    if rank == 2:
        v.set_block(2, np.array([5, 6, 3]))
    self.assertAlmostEqual(v.min(), 3)
    self.assertAlmostEqual(v.max(), 12)
    if rank == 0:
        v.set_block(0, np.array([np.inf, np.inf, np.inf, np.inf]))
    if rank == 2:
        v.set_block(2, np.array([np.inf, np.inf, np.inf]))
    self.assertEqual(v.min(), np.inf)
    self.assertEqual(v.max(), np.inf)
    v *= -1
    self.assertEqual(v.min(), -np.inf)
    self.assertEqual(v.max(), -np.inf)
    v = MPIBlockVector(3, [0, 1, 2], comm)
    v.set_block(rank, np.array([]))
    with self.assertRaisesRegex(ValueError, 'cannot get the min of a size 0 array'):
        v.min()
    with self.assertRaisesRegex(ValueError, 'cannot get the max of a size 0 array'):
        v.max()