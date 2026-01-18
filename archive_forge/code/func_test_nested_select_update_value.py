import numpy as np
import pytest
from panel.layout import GridBox, Row
from panel.pane import panel
from panel.tests.util import mpl_available
from panel.widgets import (
def test_nested_select_update_value(document, comm):
    options = {'Andrew': {'temp': [1000, 925, 700, 500, 300], 'vorticity': [500, 300]}, 'Ben': {'temp': [500, 300], 'windspeed': [700, 500, 300]}}
    levels = ['Name', 'Var', 'Level']
    value = {'Name': 'Ben', 'Var': 'temp', 'Level': 300}
    select = NestedSelect(options=options, levels=levels, value=value)
    value = {'Name': 'Ben', 'Var': 'windspeed', 'Level': 700}
    select.value = value
    assert select.options == options
    assert select.value == value
    assert select.levels == levels