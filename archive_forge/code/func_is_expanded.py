import pytest
from bokeh.models import Div
from playwright.sync_api import expect
from panel import Accordion
from panel.tests.util import serve_component
def is_expanded(card_object, card_content):
    expect(card_object.locator('svg')).to_have_class('icon icon-tabler icons-tabler-outline icon-tabler-chevron-down')
    expect(card_object).to_contain_text(card_content)
    return True