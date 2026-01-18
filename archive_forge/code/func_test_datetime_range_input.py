from datetime import date, datetime
from pathlib import Path
import numpy as np
import pytest
from bokeh.models.widgets import FileInput as BkFileInput
from panel import config
from panel.widgets import (
def test_datetime_range_input(document, comm):
    dt_input = DatetimeRangeInput(value=(datetime(2018, 1, 1), datetime(2018, 1, 3)), start=datetime(2017, 12, 31), end=datetime(2018, 1, 10), name='Datetime')
    composite = dt_input.get_root(document, comm=comm)
    label, start, end = composite.children
    assert isinstance(composite, dt_input._composite_type._bokeh_model)
    assert label.text == 'Datetime'
    assert start.value == '2018-01-01 00:00:00'
    assert end.value == '2018-01-03 00:00:00'
    dt_input._start._process_events({'value': '2018-01-01 00:00:01'})
    assert dt_input.value == (datetime(2018, 1, 1, 0, 0, 1), datetime(2018, 1, 3))
    assert label.text == 'Datetime'
    dt_input._start._process_events({'value': '2018-01-01 00:00:01a'})
    assert dt_input.value == (datetime(2018, 1, 1, 0, 0, 1), datetime(2018, 1, 3))
    assert label.text == 'Datetime (invalid)'
    dt_input._start._process_events({'value': '2018-01-11 00:00:00'})
    assert dt_input.value == (datetime(2018, 1, 1, 0, 0, 1), datetime(2018, 1, 3))
    assert label.text == 'Datetime (out of bounds)'
    dt_input._end._process_events({'value': '2018-01-11 00:00:00a'})
    assert dt_input.value == (datetime(2018, 1, 1, 0, 0, 1), datetime(2018, 1, 3))
    assert label.text == 'Datetime (out of bounds) (invalid)'
    dt_input._start._process_events({'value': '2018-01-02 00:00:00'})
    dt_input._end._process_events({'value': '2018-01-03 00:00:00'})
    assert dt_input.value == (datetime(2018, 1, 2), datetime(2018, 1, 3))
    assert label.text == 'Datetime'
    dt_input._start._process_events({'value': '2018-01-05 00:00:00'})
    assert dt_input.value == (datetime(2018, 1, 2), datetime(2018, 1, 3))
    assert label.text == 'Datetime (start of range must be <= end)'
    dt_input._start._process_events({'value': '2018-01-01 00:00:00'})
    assert dt_input.value == (datetime(2018, 1, 1), datetime(2018, 1, 3))
    assert label.text == 'Datetime'