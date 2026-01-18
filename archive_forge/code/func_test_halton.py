import numpy as np
import numpy.testing as npt
from statsmodels.tools import sequences
def test_halton():
    corners = np.array([[0, 2], [10, 5]])
    sample = sequences.halton(dim=2, n_sample=5, bounds=corners)
    out = np.array([[5.0, 3.0], [2.5, 4.0], [7.5, 2.3], [1.25, 3.3], [6.25, 4.3]])
    npt.assert_almost_equal(sample, out, decimal=1)
    sample = sequences.halton(dim=2, n_sample=3, bounds=corners, start_index=2)
    out = np.array([[7.5, 2.3], [1.25, 3.3], [6.25, 4.3]])
    npt.assert_almost_equal(sample, out, decimal=1)