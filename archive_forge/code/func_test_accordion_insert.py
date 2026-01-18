import pytest
from bokeh.models import Div
from playwright.sync_api import expect
from panel import Accordion
from panel.tests.util import serve_component
def test_accordion_insert(page, accordion_components):
    d0, d1 = accordion_components
    accordion = Accordion(d0, d1)
    serve_component(page, accordion)
    accordion_elements = page.locator('.accordion')
    expect(accordion_elements).to_have_count(len(accordion_components))
    d0_object = accordion_elements.nth(0)
    d1_object = accordion_elements.nth(1)
    expect(d0_object).to_contain_text(d0.name)
    expect(d1_object).to_contain_text(d1.name)
    inserted_div = Div(name='Inserted Div', text='Inserted text')
    accordion.insert(index=1, pane=inserted_div)
    expect(accordion_elements).to_have_count(len(accordion_components) + 1)
    d0_object = accordion_elements.nth(0)
    inserted_object = accordion_elements.nth(1)
    d1_object = accordion_elements.nth(2)
    expect(d0_object).to_contain_text(d0.name)
    expect(inserted_object).to_contain_text(inserted_div.name)
    expect(d1_object).to_contain_text(d1.name)