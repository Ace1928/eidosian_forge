import numpy as np
import scipy.sparse as sp
import cvxpy.interface as intf
from cvxpy.tests.base_test import BaseTest
def test_conversion_between_intf(self) -> None:
    """Test conversion between every pair of interfaces.
        """
    interfaces = [intf.get_matrix_interface(np.ndarray), intf.get_matrix_interface(np.matrix), intf.get_matrix_interface(sp.csc_matrix)]
    cmp_mat = [[1, 2, 3, 4], [3, 4, 5, 6], [-1, 0, 2, 4]]
    for i in range(len(interfaces)):
        for j in range(i + 1, len(interfaces)):
            intf1 = interfaces[i]
            mat1 = intf1.const_to_matrix(cmp_mat)
            intf2 = interfaces[j]
            mat2 = intf2.const_to_matrix(cmp_mat)
            for col in range(len(cmp_mat)):
                for row in range(len(cmp_mat[0])):
                    key = (slice(row, row + 1, None), slice(col, col + 1, None))
                    self.assertEqual(intf1.index(mat1, key), intf2.index(mat2, key))
                    self.assertEqual(cmp_mat[col][row], intf1.index(intf1.const_to_matrix(mat2), key))
                    self.assertEqual(intf2.index(intf2.const_to_matrix(mat1), key), cmp_mat[col][row])