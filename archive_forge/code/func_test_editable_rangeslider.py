from datetime import date, datetime
import pytest
from bokeh.models import (
from panel import config
from panel.widgets import (
@pytest.mark.parametrize('editableslider,start,end,step,val1,val2,val3,diff1', [(EditableRangeSlider, 0.1, 0.5, 0.1, (0.2, 0.4), (0.2, 0.3), (0.1, 0.5), 0.1)], ids=['EditableRangeSlider'])
def test_editable_rangeslider(document, comm, editableslider, start, end, step, val1, val2, val3, diff1):
    slider = editableslider(start=start, end=end, value=val1, name='Slider')
    widget = slider.get_root(document, comm=comm)
    assert isinstance(widget, BkColumn)
    col_items = widget.children
    assert len(col_items) == 2
    row, slider_widget = col_items
    assert slider_widget.title == ''
    assert slider_widget.step == step
    assert slider_widget.start == start
    assert slider_widget.end == end
    assert slider_widget.value == val1
    assert isinstance(row, BkRow)
    static_widget, input_widget_start, _, input_widget_end = row.children
    assert input_widget_start.title == ''
    assert input_widget_start.step == step
    assert input_widget_start.value == val1[0]
    assert input_widget_end.title == ''
    assert input_widget_end.step == step
    assert input_widget_end.value == val1[1]
    slider._process_events({'value': val2})
    assert slider.value == (input_widget_start.value, input_widget_end.value) == slider_widget.value == val2
    slider._process_events({'value_throttled': val2})
    assert slider.value_throttled == val2
    with config.set(throttled=True):
        slider._process_events({'value': val1})
        assert slider.value == val2
        slider._process_events({'value_throttled': val1})
        assert slider.value == val1
        slider.value = val3
        assert (input_widget_start.value, input_widget_end.value) == slider_widget.value == val3
    slider.name = 'New Slider'
    assert static_widget.text == 'New Slider:'
    slider.fixed_start = slider.value[0] + diff1
    assert slider._slider.start == slider.fixed_start == slider_widget.start
    slider.fixed_start = None
    slider.fixed_end = slider.value[1] - diff1
    assert slider._slider.end == slider.fixed_end == slider_widget.end
    slider.fixed_end = None