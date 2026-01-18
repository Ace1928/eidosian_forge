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
def test_global_authorize_callback(page):
    users, sessions = ([], [])

    def authorize(user_info, uri):
        users.append(user_info['user'])
        return False

    def app():
        sessions.append(state.user)
        return Markdown('Page A')
    with config.set(authorize_callback=authorize):
        _, port = serve_component(page, app, basic_auth='my_password', cookie_secret='my_secret', wait=False)
        page.locator('input[name="username"]').fill('A')
        page.locator('input[name="password"]').fill('my_password')
        page.get_by_role('button').click(force=True)
        wait_until(lambda: len(users) == 1, page)
        assert users[0] == 'A'
        assert len(sessions) == 0