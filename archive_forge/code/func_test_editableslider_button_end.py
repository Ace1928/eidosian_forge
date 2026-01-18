import pytest
from panel.tests.util import serve_component, wait_until
from panel.widgets import (
@pytest.mark.parametrize('widget', [EditableIntSlider, EditableFloatSlider], ids=['EditableIntSlider', 'EditableFloatSlider'])
def test_editableslider_button_end(page, widget):
    widget = widget(step=1)
    default_value = widget.value
    step = widget.step
    end = widget.end
    serve_component(page, widget)
    up = page.locator('button').nth(0)
    down = page.locator('button').nth(1)
    up.click()
    wait_until(lambda: widget.value == default_value + step, page)
    wait_until(lambda: widget.value == default_value + step, page)
    wait_until(lambda: widget._slider.end == end, page)
    up.click()
    wait_until(lambda: widget.value == default_value + 2 * step, page)
    wait_until(lambda: widget._slider.end == end + step, page)
    down.click()
    wait_until(lambda: widget.value == default_value + step, page)
    wait_until(lambda: widget._slider.end == end + step, page)