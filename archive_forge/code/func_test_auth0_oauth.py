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
@auth_check
def test_auth0_oauth(py_file, page):
    app = "import panel as pn; pn.pane.Markdown(pn.state.user).servable(title='A')"
    write_file(app, py_file.file)
    port = os.environ.get('AUTH0_PORT', '5701')
    cookie_secret = os.environ['OAUTH_COOKIE_SECRET']
    encryption_key = os.environ['OAUTH_ENCRYPTION_KEY']
    oauth_key = os.environ['AUTH0_OAUTH_KEY']
    oauth_secret = os.environ['AUTH0_OAUTH_SECRET']
    extra_params = os.environ['AUTH0_OAUTH_EXTRA_PARAMS']
    auth0_user = os.environ['AUTH0_OAUTH_USER']
    auth0_password = os.environ['AUTH0_OAUTH_PASSWORD']
    cmd = ['--port', port, '--oauth-provider', 'auth0', '--oauth-key', oauth_key, '--oauth-secret', oauth_secret, '--cookie-secret', cookie_secret, '--oauth-encryption-key', encryption_key, '--oauth-extra-params', extra_params, py_file.name]
    with run_panel_serve(cmd) as p:
        port = wait_for_port(p.stdout)
        page.goto(f'http://localhost:{port}')
        page.locator('input[name="username"]').fill(auth0_user)
        page.locator('input[name="password"]').fill(auth0_password)
        page.get_by_role('button', name='Continue', exact=True).click(force=True)
        expect(page.locator('.markdown')).to_have_text(auth0_user, timeout=10000)