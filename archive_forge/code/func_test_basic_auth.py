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
@pytest.mark.parametrize('prefix', ['', 'prefix'])
def test_basic_auth(py_file, page, prefix):
    app = "import panel as pn; pn.pane.Markdown(pn.state.user).servable(title='A')"
    write_file(app, py_file.file)
    app_name = os.path.basename(py_file.name)[:-3]
    cmd = ['--port', '0', '--basic-auth', 'my_password', '--cookie-secret', 'secret', py_file.name]
    if prefix:
        app_name = f'{prefix}/{app_name}'
        cmd += ['--prefix', prefix]
    with run_panel_serve(cmd) as p:
        port = wait_for_port(p.stdout)
        page.goto(f'http://localhost:{port}/{app_name}')
        page.locator('input[name="username"]').fill('test_user')
        page.locator('input[name="password"]').fill('my_password')
        page.get_by_role('button').click(force=True)
        expect(page.locator('.markdown')).to_have_text('test_user', timeout=10000)