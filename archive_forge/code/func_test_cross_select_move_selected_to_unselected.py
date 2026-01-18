import numpy as np
import pytest
from panel.layout import GridBox, Row
from panel.pane import panel
from panel.tests.util import mpl_available
from panel.widgets import (
def test_cross_select_move_selected_to_unselected():
    cross_select = CrossSelector(options=['A', 'B', 'C', 1, 2, 3], value=['A', 1], size=5)
    cross_select._lists[True].value = ['A', '1']
    cross_select._buttons[False].clicks = 1
    assert cross_select.value == []
    assert cross_select._lists[True].options == []