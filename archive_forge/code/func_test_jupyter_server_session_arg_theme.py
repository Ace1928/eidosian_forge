import sys
from pathlib import Path
import pytest
from playwright.sync_api import expect
from panel.tests.util import wait_until
def test_jupyter_server_session_arg_theme(page, jupyter_preview):
    page.goto(f'{jupyter_preview}/app.py?theme=dark', timeout=30000)
    expect(page.locator('body')).to_have_css('color', 'rgb(0, 0, 0)')