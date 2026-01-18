import numpy as np
import pytest
from panel.layout import GridBox, Row
from panel.pane import panel
from panel.tests.util import mpl_available
from panel.widgets import (
def test_multi_select(document, comm):
    select = MultiSelect(options={'A': 'A', '1': 1, 'C': object}, value=[object, 1], name='Select')
    widget = select.get_root(document, comm=comm)
    assert isinstance(widget, select._widget_type)
    assert widget.title == 'Select'
    assert widget.value == ['C', '1']
    assert widget.options == ['A', '1', 'C']
    widget.value = ['1']
    select._process_events({'value': ['1']})
    assert select.value == [1]
    widget.value = ['A', 'C']
    select._process_events({'value': ['A', 'C']})
    assert select.value == ['A', object]
    select.value = [object, 'A']
    assert widget.value == ['C', 'A']