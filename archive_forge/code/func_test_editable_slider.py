from datetime import date, datetime
import pytest
from bokeh.models import (
from panel import config
from panel.widgets import (
@pytest.mark.parametrize('editableslider,start,end,step,val1,val2,val3,diff1', [(EditableFloatSlider, 0.1, 0.5, 0.1, 0.4, 0.2, 0.5, 0.1), (EditableIntSlider, 1, 5, 1, 4, 2, 5, 1)], ids=['EditableFloatSlider', 'EditableIntSlider'])
def test_editable_slider(document, comm, editableslider, start, end, step, val1, val2, val3, diff1):
    slider = editableslider(start=start, end=end, value=val1, name='Slider')
    widget = slider.get_root(document, comm=comm)
    assert isinstance(widget, BkColumn)
    col_items = widget.children
    assert len(col_items) == 2
    row, slider_widget = col_items
    assert isinstance(slider_widget, editableslider._slider_widget._widget_type)
    assert slider_widget.title == ''
    assert slider_widget.step == step
    assert slider_widget.start == start
    assert slider_widget.end == end
    assert slider_widget.value == val1
    assert isinstance(row, BkRow)
    static_widget, input_widget = row.children
    assert isinstance(static_widget, StaticText._widget_type)
    assert static_widget.text == 'Slider:'
    assert isinstance(input_widget, editableslider._input_widget._widget_type)
    assert input_widget.title == ''
    assert input_widget.step == step
    assert input_widget.value == val1
    slider._process_events({'value': val2})
    assert slider.value == input_widget.value == slider_widget.value == val2
    slider._process_events({'value_throttled': val2})
    assert slider.value_throttled == val2
    with config.set(throttled=True):
        slider._process_events({'value': val1})
        assert slider.value == val2
        slider._process_events({'value_throttled': val1})
        assert slider.value == val1
        slider.value = val3
        assert input_widget.value == slider_widget.value == val3
    slider.name = 'New Slider'
    assert static_widget.text == 'New Slider:'
    slider.fixed_start = slider.value + diff1
    assert slider._slider.start == slider.fixed_start == slider_widget.start
    slider.fixed_start = None
    slider.fixed_end = slider.value - diff1
    assert slider._slider.end == slider.fixed_end == slider_widget.end
    slider.fixed_end = None