import pytest
from playwright.sync_api import expect
from panel import Card
from panel.tests.util import serve_component
from panel.widgets import FloatSlider, TextInput
def test_card_not_collapsible(page, card_components):
    w1, w2 = card_components
    card = Card(w1, w2, collapsible=False)
    serve_component(page, card)
    card_button = page.locator('.card-button')
    expect(card_button).to_have_count(0)
    card_elements = page.locator('.card > div, .card > button')
    expect(card_elements).to_have_count(len(card) + 1)