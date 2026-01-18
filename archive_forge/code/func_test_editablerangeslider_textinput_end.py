import pytest
from panel.tests.util import serve_component, wait_until
from panel.widgets import (
@pytest.mark.parametrize('widget,val1,val2,val3,func', [(EditableRangeSlider, 25.5, 24.5, 19.5, float)], ids=['EditableRangeSlider'])
def test_editablerangeslider_textinput_end(page, widget, val1, val2, val3, func):
    widget = widget()
    serve_component(page, widget)
    text_input = _editable_text_input(page, nth=1)
    text_input.value = val1
    wait_until(lambda: widget.value == (0, val1), page)
    wait_until(lambda: widget._slider.end == val1, page)
    wait_until(lambda: func(text_input.value) == val1, page)
    text_input.value = val2
    wait_until(lambda: widget.value == (0, val2), page)
    wait_until(lambda: widget._slider.end == val1, page)
    wait_until(lambda: func(text_input.value) == val2, page)
    widget.fixed_end = val3
    wait_until(lambda: widget.value == (0, val3), page)
    wait_until(lambda: widget._slider.end == val3, page)
    wait_until(lambda: func(text_input.value) == val3, page)