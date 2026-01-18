import numpy as np
import pytest
from panel.layout import GridBox, Row
from panel.pane import panel
from panel.tests.util import mpl_available
from panel.widgets import (
def test_nested_select_init_empty(document, comm):
    select = NestedSelect()
    assert select.value is None
    assert select.options is None
    assert select.levels == []