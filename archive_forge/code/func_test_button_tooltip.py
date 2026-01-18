import pytest
from bokeh.models import Tooltip
from playwright.sync_api import Expect, expect
from panel.tests.util import serve_component, wait_until
from panel.widgets import (
@pytest.mark.parametrize('description', ['Test', Tooltip(content='Test', position='right'), TooltipIcon(value='Test')], ids=['str', 'Tooltip', 'TooltipIcon'])
@pytest.mark.parametrize('button_fn,button_locator', [(lambda **kw: Button(**kw), '.bk-btn'), (lambda **kw: CheckButtonGroup(options=['A', 'B'], **kw), '.bk-btn-group'), (lambda **kw: RadioButtonGroup(options=['A', 'B'], **kw), '.bk-btn-group')], ids=['Button', 'CheckButtonGroup', 'RadioButtonGroup'])
def test_button_tooltip(page, button_fn, button_locator, description):
    pn_button = button_fn(name='test', description=description, description_delay=0)
    serve_component(page, pn_button)
    button = page.locator(button_locator)
    expect(button).to_have_count(1)
    tooltip = page.locator('.bk-tooltip-content')
    expect(tooltip).to_have_count(0)
    page.hover(button_locator)
    tooltip = page.locator('.bk-tooltip-content')
    expect(tooltip).to_have_count(1)
    expect(tooltip).to_have_text('Test')
    page.hover('body')
    tooltip = page.locator('.bk-tooltip-content')
    expect(tooltip).to_have_count(0)