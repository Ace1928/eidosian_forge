import numpy as np
import pytest
from matplotlib import lines, patches, text, spines, axis
from matplotlib import pyplot as plt
import cirq.testing
from cirq.vis.density_matrix import plot_density_matrix
from cirq.vis.density_matrix import _plot_element_of_density_matrix
@pytest.mark.usefixtures('closefigures')
@pytest.mark.parametrize('show_rect', [True, False])
@pytest.mark.parametrize('value', [0.0, 1.0, 0.5 + 0.3j, 0.2 + 0.1j, 0.5 + 0.5j])
def test_density_element_plot(value, show_rect):
    _, ax = plt.subplots(figsize=(10, 10))
    _plot_element_of_density_matrix(ax, 0, 0, np.abs(value), np.angle(value), show_rect=False, show_text=False)
    plotted_lines = [c for c in ax.get_children() if isinstance(c, lines.Line2D)]
    assert len(plotted_lines) == 1
    line_position = plotted_lines[0].get_xydata()
    angle = np.arctan((line_position[1, 1] - line_position[0, 1]) / (line_position[1, 0] - line_position[0, 0]))
    assert np.isclose(np.angle(value), angle)
    circles_in = [c for c in ax.get_children() if isinstance(c, patches.Circle) and c.fill]
    assert len(circles_in) == 1
    circles_out = [c for c in ax.get_children() if isinstance(c, patches.Circle) and (not c.fill)]
    assert len(circles_out) == 1
    assert np.isclose(circles_in[0].radius, circles_out[0].radius * np.abs(value))
    if show_rect:
        rectangles = [r for r in ax.get_children() if isinstance(r, patches.Rectangle)]
        assert len(rectangles) == 1
        assert rectangles[0].fill