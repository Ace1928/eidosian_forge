import numpy as np
import pytest
from panel.layout import GridBox, Row
from panel.pane import panel
from panel.tests.util import mpl_available
from panel.widgets import (
def test_nested_select_partial_options_init(document, comm):
    options = {'Ben': {}, 'Andrew': {'temp': [1000, 925, 700, 500, 300], 'vorticity': [500, 300]}}
    levels = ['Name', 'Var', 'Level']
    select = NestedSelect(options=options, levels=levels)
    assert select._widgets[0].value == 'Ben'
    assert select._widgets[1].value is None
    assert select._widgets[2].value is None
    assert select._widgets[0].visible
    assert not select._widgets[1].visible
    assert not select._widgets[2].visible
    assert select.value == {'Name': 'Ben', 'Var': None, 'Level': None}