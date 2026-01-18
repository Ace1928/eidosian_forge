import pytest
from panel.layout import Row
from panel.tests.util import serve_component, wait_until
from panel.widgets import (
def test_button_icon_on_click_method(page):

    def on_click(event):
        static_text.value = f'Clicks: {button.clicks}'
    static_text = StaticText(value='Clicks: 0')
    button = ButtonIcon()
    button.on_click(on_click)
    row = Row(button, static_text)
    serve_component(page, row)
    page.click('.bk-TablerIcon')
    wait_until(lambda: static_text.value == 'Clicks: 1', page)
    assert static_text.value == 'Clicks: 1'
    button.clicks += 1
    assert static_text.value == 'Clicks: 2'