import numpy as np
import pytest
from matplotlib import lines, patches, text, spines, axis
from matplotlib import pyplot as plt
import cirq.testing
from cirq.vis.density_matrix import plot_density_matrix
from cirq.vis.density_matrix import _plot_element_of_density_matrix
@pytest.mark.usefixtures('closefigures')
@pytest.mark.parametrize('show_text', [True, False])
@pytest.mark.parametrize('size', [2, 4, 8, 16])
def test_density_matrix_circle_rectangle_sizes(size, show_text):
    matrix = cirq.testing.random_density_matrix(size)
    ax = plot_density_matrix(matrix, show_text=show_text, title='Test Density Matrix Plot')
    circles = [c for c in ax.get_children() if isinstance(c, patches.Circle)]
    mean_radius = np.mean([c.radius for c in circles if c.fill])
    mean_value = np.mean(np.abs(matrix))
    circles = np.array(sorted(circles, key=lambda x: (x.fill, x.center[0], -x.center[1]))).reshape((2, size, size))
    for i in range(size):
        for j in range(size):
            assert np.isclose(np.abs(matrix[i, j]) * mean_radius / mean_value, circles[1, i, j].radius)
    rects = [r for r in ax.get_children() if isinstance(r, patches.Rectangle) and r.get_alpha() is not None]
    assert len(rects) == size
    mean_size = np.mean([r.get_height() for r in rects])
    mean_value = np.trace(np.abs(matrix)) / size
    rects = np.array(sorted(rects, key=lambda x: x.get_x()))
    for i in range(size):
        assert np.isclose(np.abs(matrix[i, i]) * mean_size / mean_value, rects[i].get_height())
        rect_points = rects[i].get_bbox().get_points()
        assert np.isclose((rect_points[0, 0] + rect_points[1, 0]) / 2, circles[1, i, i].center[0])
        assert np.abs((rect_points[0, 1] + rect_points[1, 1]) / 2 - circles[1, i, i].center[1]) <= circles[0, i, i].radius * 1.5