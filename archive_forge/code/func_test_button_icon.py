import pytest
from panel.layout import Row
from panel.tests.util import serve_component, wait_until
from panel.widgets import (
def test_button_icon(page):
    icon = ButtonIcon(icon='clipboard', active_icon='check', toggle_duration=2000, size='5em')
    serve_component(page, icon)
    assert icon.icon == 'clipboard'
    assert not icon.disabled
    assert page.locator('.clipboard')
    events = []

    def cb(event):
        events.append(event)
    icon.param.watch(cb, 'clicks')
    page.click('.bk-TablerIcon')
    wait_until(lambda: len(events) == 1, page)
    assert len(events) == 1
    assert icon.clicks == 1
    assert page.locator('.check')
    wait_until(lambda: not icon.disabled, page)
    assert not icon.disabled
    assert page.locator('.clipboard')