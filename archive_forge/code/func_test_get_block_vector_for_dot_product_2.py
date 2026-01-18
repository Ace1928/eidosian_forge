import warnings
import pyomo.common.unittest as unittest
from pyomo.contrib.pynumero.dependencies import (
def test_get_block_vector_for_dot_product_2(self):
    rank = comm.Get_rank()
    rank_ownership = np.array([[1, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]])
    m = MPIBlockMatrix(4, 3, rank_ownership, comm)
    sub_m = np.array([[1, 0], [0, 1]])
    sub_m = coo_matrix(sub_m)
    if rank == 0:
        m.set_block(3, rank, sub_m.copy())
    elif rank == 1:
        m.set_block(0, 0, sub_m.copy())
        m.set_block(rank, rank, sub_m.copy())
        m.set_block(3, rank, sub_m.copy())
    else:
        m.set_block(rank, rank, sub_m.copy())
        m.set_block(3, rank, sub_m.copy())
    rank_ownership = np.array([-1, 1, 2])
    v = MPIBlockVector(3, rank_ownership, comm)
    sub_v = np.ones(2)
    v.set_block(0, sub_v.copy())
    if rank != 0:
        v.set_block(rank, sub_v.copy())
    res = m._get_block_vector_for_dot_product(v)
    self.assertIs(res, v)