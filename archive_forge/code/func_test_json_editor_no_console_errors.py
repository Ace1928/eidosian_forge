import sys
import pytest
from playwright.sync_api import expect
from panel.tests.util import serve_component, wait_until
from panel.widgets import JSONEditor
def test_json_editor_no_console_errors(page):
    editor = JSONEditor(value={'str': 'string', 'int': 1})
    msgs, _ = serve_component(page, editor)
    expect(page.locator('.jsoneditor')).to_have_count(1)
    assert [msg for msg in msgs if msg.type == 'error' and 'favicon' not in msg.location['url']] == []