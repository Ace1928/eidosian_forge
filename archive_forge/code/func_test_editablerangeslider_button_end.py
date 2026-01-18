import pytest
from panel.tests.util import serve_component, wait_until
from panel.widgets import (
@pytest.mark.parametrize('widget', [EditableRangeSlider], ids=['EditableRangeSlider'])
def test_editablerangeslider_button_end(page, widget):
    widget = widget(step=1)
    default_value = widget.value
    step = widget.step
    end = widget.end
    serve_component(page, widget)
    up = page.locator('button').nth(2)
    down = page.locator('button').nth(3)
    up.click()
    wait_until(lambda: widget.value == (0, default_value[1] + step), page)
    wait_until(lambda: widget._slider.end == end + step, page)
    up.click()
    wait_until(lambda: widget.value == (0, default_value[1] + 2 * step), page)
    wait_until(lambda: widget._slider.end == end + 2 * step, page)
    down.click()
    wait_until(lambda: widget.value == (0, default_value[1] + step), page)
    wait_until(lambda: widget._slider.end == end + 2 * step, page)