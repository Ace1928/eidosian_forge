import numpy as np
import pytest
from panel.layout import GridBox, Row
from panel.pane import panel
from panel.tests.util import mpl_available
from panel.widgets import (
def test_nested_select_update_levels_invalid(document, comm):
    options = {'Andrew': {'temp': [1000, 925, 700, 500, 300], 'vorticity': [500, 300]}, 'Ben': {'temp': [500, 300], 'windspeed': [700, 500, 300]}}
    value = {'Name': 'Ben', 'Var': 'temp', 'Level': 300}
    select = NestedSelect(options=options, levels=['Name', 'Var', 'Level'], value=value)
    levels = ['user', 'wx_var', 'lev', 'abc']
    with pytest.raises(ValueError, match='must be of length 3'):
        select.levels = levels