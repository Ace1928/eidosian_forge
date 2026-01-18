import numpy as np # noqa F403
import scipy.sparse as spar
from cvxpy.tests.base_test import BaseTest
from cvxpy.utilities import linalg as lau
def check_factor(self, L, places=5):
    diag = L.diagonal()
    self.assertTrue(np.all(diag > 0))
    delta = (L - spar.tril(L)).toarray().flatten()
    self.assertItemsAlmostEqual(delta, np.zeros(delta.size), places)