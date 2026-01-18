import numpy as np
import pytest
from panel.layout import GridBox, Row
from panel.pane import panel
from panel.tests.util import mpl_available
from panel.widgets import (
def test_nested_select_update_all(document, comm):
    options = {'Andrew': {'temp': [1000, 925, 700, 500, 300], 'vorticity': [500, 300]}, 'Ben': {'temp': [500, 300], 'windspeed': [700, 500, 300]}}
    value = {'Name': 'Ben', 'Var': 'temp', 'Level': 300}
    select = NestedSelect(options=options, levels=['Name', 'Var', 'Level'], value=value)
    new_levels = ['N', 'V', 'L']
    new_options = {'Ben': {'temp': [500, 300], 'windspeed': [1000]}}
    new_value = {'N': 'Ben', 'V': 'windspeed', 'L': 1000}
    select.param.update(options=new_options, levels=new_levels, value=new_value)
    assert select.options == new_options
    assert select.value == new_value
    assert select.levels == new_levels