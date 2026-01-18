from numpy.testing import (
from numpy import (
import numpy as np
import pytest
def test_fortran_order(self):
    vals = array(100 * get_mat(5) + 1, order='F', dtype='l')
    self.test_matrix(vals)