from datetime import date, datetime
from pathlib import Path
import numpy as np
import pytest
from bokeh.models.widgets import FileInput as BkFileInput
from panel import config
from panel.widgets import (
def test_daterange_picker(document, comm):
    date_range_picker = DateRangePicker(name='DateRangePicker', value=(date(2018, 9, 2), date(2018, 9, 3)), start=date(2018, 9, 1), end=date(2018, 9, 10))
    widget = date_range_picker.get_root(document, comm=comm)
    assert isinstance(widget, date_range_picker._widget_type)
    assert widget.title == 'DateRangePicker'
    assert widget.value == ('2018-09-02', '2018-09-03')
    assert widget.min_date == '2018-09-01'
    assert widget.max_date == '2018-09-10'
    date_range_picker._process_events({'value': ('2018-09-03', '2018-09-04')})
    assert date_range_picker.value == (date(2018, 9, 3), date(2018, 9, 4))
    date_range_picker._process_events({'value': ('2018-09-05', '2018-09-08')})
    assert date_range_picker.value == (date(2018, 9, 5), date(2018, 9, 8))
    value = date_range_picker._process_param_change({'value': (date(2018, 9, 4), date(2018, 9, 5))})
    assert value['value'] == ('2018-09-04', '2018-09-05')
    value = date(2018, 9, 4)
    assert date_range_picker._convert_date_to_string(value) == '2018-09-04'
    assert date_range_picker._convert_string_to_date(date_range_picker._convert_date_to_string(value)) == value
    with pytest.raises(ValueError):
        date_range_picker._process_events({'value': ('2018-08-31', '2018-09-01')})
    with pytest.raises(ValueError):
        date_range_picker._process_events({'value': ('2018-09-10', '2018-09-11')})