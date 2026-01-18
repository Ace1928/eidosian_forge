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
def test_density_matrix_plotter(size, show_text):
    matrix = cirq.testing.random_density_matrix(size)
    ax = plot_density_matrix(matrix, show_text=show_text, title='Test Density Matrix Plot')
    assert ax.get_title() == 'Test Density Matrix Plot'
    for obj in ax.get_children():
        assert isinstance(obj, (patches.Circle, spines.Spine, axis.XAxis, axis.YAxis, lines.Line2D, patches.Rectangle, text.Text))