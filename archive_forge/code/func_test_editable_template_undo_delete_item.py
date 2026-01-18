import pytest
from playwright.sync_api import expect
from panel.pane import Markdown
from panel.template import EditableTemplate
from panel.tests.util import serve_component, wait_until
def test_editable_template_undo_delete_item(page):
    tmpl = EditableTemplate()
    md1 = Markdown('1')
    md2 = Markdown('2')
    tmpl.main[:] = [md1, md2]
    serve_component(page, tmpl)
    page.locator('.muuri-handle.delete').nth(1).click()
    wait_until(lambda: tmpl.layout.get(id(md2), {}).get('visible') == False, page)
    page.locator('#grid-undo').click()
    wait_until(lambda: tmpl.layout.get(id(md2), {}).get('visible') == True, page)