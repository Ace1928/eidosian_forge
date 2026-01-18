import pytest
from bokeh.models import Div
from playwright.sync_api import expect
from panel import Accordion
from panel.tests.util import serve_component
def test_accordion_default(page, accordion_components):
    d0, d1 = accordion_components
    accordion = Accordion(d0, d1)
    serve_component(page, accordion)
    accordion_elements = page.locator('.accordion')
    expect(accordion_elements).to_have_count(len(accordion_components))
    d0_object = accordion_elements.nth(0)
    d1_object = accordion_elements.nth(1)
    d0_object.wait_for()
    d1_object.wait_for()
    expect(d0_object).to_contain_text(d0.name)
    expect(d1_object).to_contain_text(d1.name)
    assert is_collapsed(card_object=d0_object, card_content=d0.text)
    assert is_collapsed(card_object=d1_object, card_content=d1.text)
    d0_object.click()
    d1_object.click()
    assert is_expanded(card_object=d0_object, card_content=d0.text)
    assert is_expanded(card_object=d1_object, card_content=d1.text)