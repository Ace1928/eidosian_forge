import pytest
from panel.tests.util import serve_component, wait_until
from panel.widgets import (
def test_editablerangeslider_no_overlap(page):
    widget = EditableRangeSlider(value=(0, 2), step=1)
    serve_component(page, widget)
    up_start = page.locator('button').nth(0)
    down_start = page.locator('button').nth(1)
    down_end = page.locator('button').nth(3)
    up_start.click(click_count=3)
    wait_until(lambda: widget.value == (2, 2), page)
    down_start.click()
    wait_until(lambda: widget.value == (1, 2), page)
    down_end.click(click_count=3)
    wait_until(lambda: widget.value == (1, 1), page)