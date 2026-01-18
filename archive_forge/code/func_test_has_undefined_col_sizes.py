import pyomo.common.unittest as unittest
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as sp
from scipy.sparse import coo_matrix, bmat
from pyomo.contrib.pynumero.sparse import (
import warnings
def test_has_undefined_col_sizes(self):
    self.assertFalse(self.basic_m.has_undefined_col_sizes())