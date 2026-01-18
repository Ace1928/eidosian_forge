import sys
from pathlib import Path
import pytest
from playwright.sync_api import expect
from panel.tests.util import wait_until
def test_jupyter_server_kernel_error(page, jupyter_preview):
    page.goto(f'{jupyter_preview}/app.py?kernel=blah', timeout=30000)
    expect(page.locator('#error-type')).to_have_text('Kernel Error')
    expect(page.locator('#error')).to_have_text("No such kernel 'blah'")
    page.select_option('select#kernel-select', 'python3')
    wait_until(lambda: page.text_content('pre') == '0', page)
    page.click('button')
    wait_until(lambda: page.text_content('pre') == '1', page)