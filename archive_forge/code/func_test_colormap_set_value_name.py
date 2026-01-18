import numpy as np
import pytest
from panel.layout import GridBox, Row
from panel.pane import panel
from panel.tests.util import mpl_available
from panel.widgets import (
def test_colormap_set_value_name(document, comm):
    color_map = ColorMap(options={'A': ['#ff0', '#0ff'], 'B': ['#00f', '#f00']}, value=['#00f', '#f00'])
    model = color_map.get_root(document, comm=comm)
    assert model.value == 'B'
    assert color_map.value_name == 'B'
    color_map.value = ['#ff0', '#0ff']
    assert model.value == 'A'
    assert color_map.value_name == 'A'