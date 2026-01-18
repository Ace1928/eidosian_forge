import pytest
from playwright.sync_api import expect
from panel.pane import Markdown
from panel.template import EditableTemplate
from panel.tests.util import serve_component, wait_until
def test_editable_template_visible(page):
    md1 = Markdown('1')
    md2 = Markdown('2')
    tmpl = EditableTemplate(layout={id(md2): {'width': 50, 'height': 50, 'visible': False}})
    tmpl.main[:] = [md1, md2]
    serve_component(page, tmpl)
    md2_item = page.locator('.muuri-grid-item').nth(1)
    expect(md2_item).to_have_class('muuri-grid-item muuri-item-hidden muuri-item')