import pytest
from playwright.sync_api import expect
from panel.pane import Markdown
from panel.template import MaterialTemplate
from panel.tests.util import serve_component
def test_material_template_no_console_errors(page):
    tmpl = MaterialTemplate()
    md = Markdown('Initial')
    tmpl.main.append(md)
    msgs, _ = serve_component(page, tmpl)
    expect(page.locator('.markdown').locator('div')).to_have_text('Initial\n')
    assert [msg for msg in msgs if msg.type == 'error'] == []