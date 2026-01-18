import os
import pathlib
import time
import pytest
from playwright.sync_api import expect
from panel.config import config
from panel.io.state import state
from panel.pane import Markdown
from panel.tests.util import (
@linux_only
def test_authorize_callback_redirect(page):

    def authorize(user_info, uri):
        if uri == '/':
            user = user_info['user']
            if user == 'A':
                return '/a'
            elif user == 'B':
                return '/b'
            else:
                return False
        return True
    app1 = Markdown('Page A')
    app2 = Markdown('Page B')
    with config.set(authorize_callback=authorize):
        _, port = serve_component(page, {'a': app1, 'b': app2, '/': 'Index'}, basic_auth='my_password', cookie_secret='my_secret', wait=False)
        page.locator('input[name="username"]').fill('A')
        page.locator('input[name="password"]').fill('my_password')
        page.get_by_role('button').click(force=True)
        expect(page.locator('.markdown').locator('div')).to_have_text('Page A\n')
        page.goto(f'http://localhost:{port}/logout')
        page.goto(f'http://localhost:{port}/login')
        page.locator('input[name="username"]').fill('B')
        page.locator('input[name="password"]').fill('my_password')
        page.get_by_role('button').click(force=True)
        expect(page.locator('.markdown').locator('div')).to_have_text('Page B\n')