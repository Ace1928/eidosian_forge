import numpy as np
import pytest
from panel.layout import GridBox, Row
from panel.pane import panel
from panel.tests.util import mpl_available
from panel.widgets import (
@mpl_available
def test_colormap_mpl_cmap(document, comm):
    from matplotlib.cm import Set1, tab10
    color_map = ColorMap(options={'tab10': tab10, 'Set1': Set1}, value_name='Set1')
    model = color_map.get_root(document, comm=comm)
    assert model.items == [('tab10', ['rgba(31, 119, 180, 1)', 'rgba(255, 127, 14, 1)', 'rgba(44, 160, 44, 1)', 'rgba(214, 39, 40, 1)', 'rgba(148, 103, 189, 1)', 'rgba(140, 86, 75, 1)', 'rgba(227, 119, 194, 1)', 'rgba(127, 127, 127, 1)', 'rgba(188, 189, 34, 1)', 'rgba(23, 190, 207, 1)']), ('Set1', ['rgba(228, 26, 28, 1)', 'rgba(55, 126, 184, 1)', 'rgba(77, 175, 74, 1)', 'rgba(152, 78, 163, 1)', 'rgba(255, 127, 0, 1)', 'rgba(255, 255, 51, 1)', 'rgba(166, 86, 40, 1)', 'rgba(247, 129, 191, 1)', 'rgba(153, 153, 153, 1)'])]