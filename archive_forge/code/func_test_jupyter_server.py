import sys
from pathlib import Path
import pytest
from playwright.sync_api import expect
from panel.tests.util import wait_until
def test_jupyter_server(page, jupyter_preview):
    page.goto(f'{jupyter_preview}/app.py', timeout=30000)
    assert page.text_content('pre') == '0'
    page.click('button')
    wait_until(lambda: page.text_content('pre') == '1', page)
    page.click('button')
    wait_until(lambda: page.text_content('pre') == '2', page)