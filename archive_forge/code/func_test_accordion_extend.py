import pytest
from bokeh.models import Div
from playwright.sync_api import expect
from panel import Accordion
from panel.tests.util import serve_component
def test_accordion_extend(page, accordion_components):
    d0, d1 = accordion_components
    accordion = Accordion(d0, d1)
    serve_component(page, accordion)
    accordion_elements = page.locator('.accordion')
    expect(accordion_elements).to_have_count(len(accordion_components))
    d2 = Div(name='Div 2', text='Text 2')
    additional_list = [d2]
    accordion.extend(additional_list)
    expect(accordion_elements).to_have_count(len(accordion_components) + len(additional_list))
    d0_object = accordion_elements.nth(0)
    d1_object = accordion_elements.nth(1)
    d2_object = accordion_elements.nth(2)
    assert is_collapsed(card_object=d0_object, card_content=d0.text)
    assert is_collapsed(card_object=d1_object, card_content=d1.text)
    assert is_collapsed(card_object=d2_object, card_content=d2.text)