from datetime import date, datetime
from pathlib import Path
import numpy as np
import pytest
from bokeh.models.widgets import FileInput as BkFileInput
from panel import config
from panel.widgets import (
def test_datetime_input(document, comm):
    dt_input = DatetimeInput(value=datetime(2018, 1, 1), start=datetime(2017, 12, 31), end=datetime(2018, 1, 10), name='Datetime')
    widget = dt_input.get_root(document, comm=comm)
    assert isinstance(widget, dt_input._widget_type)
    assert widget.title == 'Datetime'
    assert widget.value == '2018-01-01 00:00:00'
    dt_input._process_events({'value': '2018-01-01 00:00:01'})
    assert dt_input.value == datetime(2018, 1, 1, 0, 0, 1)
    assert widget.title == 'Datetime'
    dt_input._process_events({'value': '2018-01-01 00:00:01a'})
    assert dt_input.value == datetime(2018, 1, 1, 0, 0, 1)
    assert widget.title == 'Datetime (invalid)'
    dt_input._process_events({'value': '2018-01-11 00:00:00'})
    assert dt_input.value == datetime(2018, 1, 1, 0, 0, 1)
    assert widget.title == 'Datetime (out of bounds)'
    dt_input._process_events({'value': '2018-01-02 00:00:01'})
    assert dt_input.value == datetime(2018, 1, 2, 0, 0, 1)
    assert widget.title == 'Datetime'
    with pytest.raises(ValueError):
        dt_input.value = datetime(2017, 12, 30)