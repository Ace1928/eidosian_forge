from datetime import date, datetime
import pytest
from bokeh.models import (
from panel import config
from panel.widgets import (
def test_date_slider(document, comm):
    date_slider = DateSlider(name='DateSlider', value=date(2018, 9, 4), start=date(2018, 9, 1), end=date(2018, 9, 10))
    widget = date_slider.get_root(document, comm=comm)
    assert isinstance(widget, date_slider._widget_type)
    assert widget.title == 'DateSlider'
    assert widget.value == 1536019200000
    assert widget.start == 1535760000000.0
    assert widget.end == 1536537600000.0
    epoch = datetime(1970, 1, 1)
    widget.value = (datetime(2018, 9, 3) - epoch).total_seconds() * 1000
    date_slider._process_events({'value': widget.value})
    assert date_slider.value == date(2018, 9, 3)
    date_slider._process_events({'value_throttled': (datetime(2018, 9, 3) - epoch).total_seconds() * 1000})
    assert date_slider.value_throttled == date(2018, 9, 3)
    date_slider._process_events({'value': (datetime(2018, 9, 4) - epoch).total_seconds() * 1000.0})
    assert date_slider.value == date(2018, 9, 4)
    date_slider._process_events({'value_throttled': (datetime(2018, 9, 4) - epoch).total_seconds() * 1000.0})
    assert date_slider.value_throttled == date(2018, 9, 4)
    date_slider.value = date(2018, 9, 6)
    assert widget.value == 1536192000000
    epoch_time = lambda dt: (dt - epoch).total_seconds() * 1000
    with config.set(throttled=True):
        date_slider._process_events({'value': epoch_time(datetime(2021, 5, 15))})
        assert date_slider.value == date(2018, 9, 6)
        date_slider._process_events({'value_throttled': epoch_time(datetime(2021, 5, 15))})
        assert date_slider.value == date(2021, 5, 15)
        date_slider.value = date(2021, 5, 12)
        assert widget.value == 1620777600000