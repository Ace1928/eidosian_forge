import pytest
from playwright.sync_api import expect
from panel import Card
from panel.tests.util import serve_component
from panel.widgets import FloatSlider, TextInput
def test_card_objects(page, card_components):
    w1, w2 = card_components
    card = Card(w1, w2)
    serve_component(page, card)
    card.objects = [w2]
    card_elements = page.locator('.card > div, .card > button')
    expect(card_elements).to_have_count(2)
    card_header = card_elements.nth(0)
    w2_object = card_elements.nth(1)
    expect(card_header).to_have_class('card-header')
    expect(w2_object).to_have_class('bk-TextInput class_w2')
    w3 = TextInput(name='Text:', css_classes=['class_w3'])
    card.append(w3)
    expect(card_elements).to_have_count(3)
    expect(card_elements.nth(2)).to_have_class('bk-TextInput class_w3')