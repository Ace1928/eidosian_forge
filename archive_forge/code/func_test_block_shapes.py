import warnings
import pyomo.common.unittest as unittest
from pyomo.contrib.pynumero.dependencies import (
def test_block_shapes(self):
    m, n = self.square_mpi_mat.bshape
    mpi_shapes = self.square_mpi_mat.block_shapes()
    serial_shapes = self.square_serial_mat.block_shapes()
    for i in range(m):
        for j in range(n):
            self.assertEqual(serial_shapes[i][j], mpi_shapes[i][j])