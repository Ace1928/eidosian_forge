import numpy as np
import skimage.graph.mcp as mcp
from skimage._shared.testing import assert_array_equal, assert_almost_equal, parametrize
from skimage._shared._warnings import expected_warnings
def test_no_diagonal():
    with expected_warnings(['Upgrading NumPy' + warning_optional]):
        m = mcp.MCP(a, fully_connected=False)
    costs, traceback = m.find_costs([(1, 6)])
    return_path = m.traceback((7, 2))
    assert_array_equal(costs, [[2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0], [1.0, 0.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0], [1.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 4.0], [1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 5.0], [1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
    assert_array_equal(return_path, [(1, 6), (1, 5), (1, 4), (1, 3), (1, 2), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (7, 2)])