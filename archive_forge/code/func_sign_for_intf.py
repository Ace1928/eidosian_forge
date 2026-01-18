import numpy as np
import scipy.sparse as sp
import cvxpy.interface as intf
from cvxpy.tests.base_test import BaseTest
def sign_for_intf(self, interface) -> None:
    """Test sign for a given interface.
        """
    mat = interface.const_to_matrix([[1, 2, 3, 4], [3, 4, 5, 6]])
    self.assertEqual(intf.sign(mat), (True, False))
    self.assertEqual(intf.sign(-mat), (False, True))
    self.assertEqual(intf.sign(0 * mat), (True, True))
    mat = interface.const_to_matrix([[-1, 2, 3, 4], [3, 4, 5, 6]])
    self.assertEqual(intf.sign(mat), (False, False))