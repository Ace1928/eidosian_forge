import pytest
from playwright.sync_api import expect
from panel import Card
from panel.tests.util import serve_component
from panel.widgets import FloatSlider, TextInput
def test_card_custom_css(page):
    additional_css_class = 'css_class'
    additional_header_css_class = 'header_css_class'
    additional_button_css_class = 'button_css_class'
    card = Card()
    card.css_classes.append(additional_css_class)
    card.header_css_classes.append(additional_header_css_class)
    card.button_css_classes.append(additional_button_css_class)
    serve_component(page, card)
    card_widget = page.locator(f'.card.{additional_css_class}')
    expect(card_widget).to_have_count(1)
    card_header = page.locator(f'.card-header.{additional_header_css_class}')
    expect(card_header).to_have_count(1)
    card_button = page.locator(f'.card-button.{additional_button_css_class}')
    expect(card_button).to_have_count(1)