import pytest
from playwright.sync_api import expect
from panel import Card
from panel.tests.util import serve_component
from panel.widgets import FloatSlider, TextInput
def test_card_default(page, card_components):
    w1, w2 = card_components
    card = Card(w1, w2)
    serve_component(page, card)
    card_elements = page.locator('.card > div, .card > button')
    expect(card_elements).to_have_count(len(card) + 1)
    card_header = card_elements.nth(0)
    w1_object = card_elements.nth(1)
    w2_object = card_elements.nth(2)
    assert 'card-header' in card_header.get_attribute('class')
    assert 'class_w1' in w1_object.get_attribute('class')
    assert 'class_w2' in w2_object.get_attribute('class')
    card_button = page.locator('.card-button')
    expect(card_button.locator('svg')).to_have_class('icon icon-tabler icons-tabler-outline icon-tabler-chevron-down')