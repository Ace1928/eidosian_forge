from datetime import date, datetime
from pathlib import Path
import numpy as np
import pytest
from bokeh.models.widgets import FileInput as BkFileInput
from panel import config
from panel.widgets import (
def test_literal_input(document, comm):
    literal = LiteralInput(value={}, type=dict, name='Literal')
    widget = literal.get_root(document, comm=comm)
    assert isinstance(widget, literal._widget_type)
    assert widget.title == 'Literal'
    assert widget.value == '{}'
    literal._process_events({'value': "{'key': (0, 2)}"})
    assert literal.value == {'key': (0, 2)}
    assert widget.title == 'Literal'
    literal._process_events({'value': '(0, 2)'})
    assert literal.value == {'key': (0, 2)}
    assert widget.title == 'Literal (wrong type)'
    literal._process_events({'value': 'invalid'})
    assert literal.value == {'key': (0, 2)}
    assert widget.title == 'Literal (invalid)'
    literal._process_events({'value': "{'key': (0, 3)}"})
    assert literal.value == {'key': (0, 3)}
    assert widget.title == 'Literal'
    with pytest.raises(ValueError):
        literal.value = []