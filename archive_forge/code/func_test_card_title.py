import pytest
from playwright.sync_api import expect
from panel import Card
from panel.tests.util import serve_component
from panel.widgets import FloatSlider, TextInput
def test_card_title(page, card_components):
    w1, w2 = card_components
    card_title = 'Card Title'
    card = Card(w1, w2, title=card_title)
    serve_component(page, card)
    expect(page.locator('.card-title').locator('div')).to_have_text(card_title)