import numpy as np
from skimage.graph._graph import pixel_graph, central_pixel
def test_no_mask_with_edge_func():
    """Ensure function `pixel_graph` runs when passing `edge_function` but not `mask`."""
    image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def func(x, y, z):
        return np.abs(x - y) * 0.5
    expected_g = np.array([[0.0, 1.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 3.0, 0.0, 0.0], [0.0, 3.0, 0.0, 1.0, 0.0, 1.0, 0.0, 3.0, 0.0], [0.0, 0.0, 3.0, 0.0, 1.0, 0.0, 0.0, 0.0, 3.0], [0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 1.0, 0.0]]) * 0.5
    g, n = pixel_graph(image, edge_function=func)
    np.testing.assert_array_equal(n, np.arange(image.size))
    np.testing.assert_array_equal(g.todense(), expected_g)