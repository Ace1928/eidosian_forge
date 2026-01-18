import warnings
import pyomo.common.unittest as unittest
from pyomo.contrib.pynumero.dependencies import (
def test_matvec_1(self):
    rank = comm.Get_rank()
    np.random.seed(0)
    orig_m = np.zeros((8, 8))
    for ndx in range(4):
        start = ndx * 2
        stop = (ndx + 1) * 2
        orig_m[start:stop, start:stop] = np.random.uniform(-10, 10, size=(2, 2))
        orig_m[start:stop, 6:8] = np.random.uniform(-10, 10, size=(2, 2))
        orig_m[6:8, start:stop] = np.random.uniform(-10, 10, size=(2, 2))
    orig_m[6:8, 6:8] = np.random.uniform(-10, 10, size=(2, 2))
    orig_v = np.random.uniform(-10, 10, size=8)
    correct_res = coo_matrix(orig_m) * orig_v
    rank_ownership = np.array([[0, -1, -1, 0], [-1, 1, -1, 1], [-1, -1, 2, 2], [0, 1, 2, -1]])
    m = MPIBlockMatrix(4, 4, rank_ownership, comm)
    start = rank * 2
    stop = (rank + 1) * 2
    m.set_block(rank, rank, coo_matrix(orig_m[start:stop, start:stop]))
    m.set_block(rank, 3, coo_matrix(orig_m[start:stop, 6:8]))
    m.set_block(3, rank, coo_matrix(orig_m[6:8, start:stop]))
    m.set_block(3, 3, coo_matrix(orig_m[6:8, 6:8]))
    rank_ownership = np.array([0, 1, 2, -1])
    v = MPIBlockVector(4, rank_ownership, comm)
    v.set_block(rank, orig_v[start:stop])
    v.set_block(3, orig_v[6:8])
    res: MPIBlockVector = m.dot(v)
    self.assertTrue(np.allclose(correct_res, res.make_local_copy().flatten()))
    self.assertIsInstance(res, MPIBlockVector)
    self.assertTrue(np.allclose(res.get_block(rank), correct_res[start:stop]))
    self.assertTrue(np.allclose(res.get_block(3), correct_res[6:8]))
    self.assertTrue(np.allclose(res.rank_ownership, np.array([0, 1, 2, -1])))
    self.assertFalse(res.has_none)