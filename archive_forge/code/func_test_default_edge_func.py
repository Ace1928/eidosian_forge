import numpy as np
from skimage.graph._graph import pixel_graph, central_pixel
def test_default_edge_func():
    g, n = pixel_graph(image, spacing=np.array([0.78, 0.78]))
    num_edges = len(g.data) // 2
    assert num_edges == 12
    np.testing.assert_almost_equal(g[0, 1], 0.78 * np.abs(image[0, 0] - image[0, 1]))
    np.testing.assert_array_equal(n, np.arange(image.size))