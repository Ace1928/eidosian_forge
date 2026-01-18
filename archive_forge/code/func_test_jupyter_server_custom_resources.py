import sys
from pathlib import Path
import pytest
from playwright.sync_api import expect
from panel.tests.util import wait_until
def test_jupyter_server_custom_resources(page, jupyter_preview):
    page.goto(f'{jupyter_preview}/app.py', timeout=30000)
    assert page.locator('.bk-Row').evaluate("(element) =>\n        window.getComputedStyle(element).getPropertyValue('background-color')") == 'rgb(128, 0, 128)'