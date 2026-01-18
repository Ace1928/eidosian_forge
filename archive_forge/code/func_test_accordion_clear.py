import pytest
from bokeh.models import Div
from playwright.sync_api import expect
from panel import Accordion
from panel.tests.util import serve_component
def test_accordion_clear(page, accordion_components):
    d0, d1 = accordion_components
    accordion = Accordion(d0, d1)
    serve_component(page, accordion)
    accordion_elements = page.locator('.accordion')
    expect(accordion_elements).to_have_count(len(accordion_components))
    accordion.clear()
    expect(accordion_elements).to_have_count(0)