import unittest
import numpy as np
from pygsp import graphs, filters
def test_approximations(self):
    """
        Test that the different methods for filter analysis, i.e. 'exact',
        'cheby', and 'lanczos', produce the same output.
        """
    f = filters.Heat(self._G)
    c_exact = f.filter(self._signal, method='exact')
    c_cheby = f.filter(self._signal, method='chebyshev')
    np.testing.assert_allclose(c_exact, c_cheby)
    self.assertRaises(ValueError, f.filter, self._signal, method='lanczos')