import numpy as np
import skimage.graph.mcp as mcp
from skimage._shared.testing import assert_array_equal
def test_flexible():
    mcp = FlexibleMCP(a)
    costs, traceback = mcp.find_costs([(0, 0)])
    assert_array_equal(costs[:4, :4], [[1, 2, 3, 4], [2, 2, 3, 4], [3, 3, 3, 4], [4, 4, 4, 4]])
    assert np.all(costs[-2:, :] == np.inf)
    assert np.all(costs[:, -2:] == np.inf)