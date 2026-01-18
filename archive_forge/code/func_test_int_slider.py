from datetime import date, datetime
import pytest
from bokeh.models import (
from panel import config
from panel.widgets import (
def test_int_slider(document, comm):
    slider = IntSlider(start=0, end=3, value=1, name='Slider')
    widget = slider.get_root(document, comm=comm)
    assert isinstance(widget, slider._widget_type)
    assert widget.title == 'Slider'
    assert widget.step == 1
    assert widget.start == 0
    assert widget.end == 3
    assert widget.value == 1
    slider._process_events({'value': 2})
    assert slider.value == 2
    slider._process_events({'value_throttled': 2})
    assert slider.value_throttled == 2
    slider.value = 0
    assert widget.value == 0
    slider_2 = IntSlider(start=1, end=3, name='Slider_2')
    widget_2 = slider_2.get_root(document, comm=comm)
    assert widget_2.value == widget_2.start
    with config.set(throttled=True):
        slider._process_events({'value': 1})
        assert slider.value == 0
        slider._process_events({'value_throttled': 1})
        assert slider.value == 1
        slider.value = 2
        assert widget.value == 2