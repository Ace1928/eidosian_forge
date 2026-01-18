from html import escape
import pytest
from playwright.sync_api import expect
from panel.models import HTML
from panel.pane import Markdown
from panel.tests.util import serve_component, wait_until
def test_html_model_no_stylesheet(page):
    text = '<h1>Header</h1>'
    html = HTML(text=escape(text), width=200, height=200)
    serve_component(page, html)
    header_element = page.locator('h1:has-text("Header")')
    assert header_element.is_visible()
    assert header_element.text_content() == 'Header'