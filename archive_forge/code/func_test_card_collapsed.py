import pytest
from playwright.sync_api import expect
from panel import Card
from panel.tests.util import serve_component
from panel.widgets import FloatSlider, TextInput
def test_card_collapsed(page, card_components):
    w1, w2 = card_components
    card = Card(w1, w2)
    serve_component(page, card)
    card_elements = page.locator('.card > div, .card > button')
    card_button = page.locator('.card-button')
    card_button.wait_for()
    card_button.click()
    expect(card_button.locator('svg')).to_have_class('icon icon-tabler icons-tabler-outline icon-tabler-chevron-right')
    expect(card_elements).to_have_count(1)
    card_button.wait_for()
    card_button.click()
    expect(card_elements).to_have_count(len(card) + 1)
    expect(card_button.locator('svg')).to_have_class('icon icon-tabler icons-tabler-outline icon-tabler-chevron-down')