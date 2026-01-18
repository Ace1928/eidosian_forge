import numpy as np
import pytest
from panel.layout import GridBox, Row
from panel.pane import panel
from panel.tests.util import mpl_available
from panel.widgets import (
def test_nested_select_update_options(document, comm):
    options = {'Andrew': {'temp': [1000, 925, 700, 500, 300], 'vorticity': [500, 300]}, 'Ben': {'temp': [500, 300], 'windspeed': [700, 500, 300]}}
    levels = ['Name', 'Var', 'Level']
    value = {'Level': 1000, 'Name': 'Andrew', 'Var': 'temp'}
    select = NestedSelect(options=options, levels=levels, value=value)
    options = {'August': {'temp': [500, 300]}}
    select.options = options
    assert select.options == options
    assert select.value == {'Level': 500, 'Name': 'August', 'Var': 'temp'}
    assert select.levels == levels