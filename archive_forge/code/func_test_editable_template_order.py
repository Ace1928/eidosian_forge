import pytest
from playwright.sync_api import expect
from panel.pane import Markdown
from panel.template import EditableTemplate
from panel.tests.util import serve_component, wait_until
def test_editable_template_order(page):
    md1 = Markdown('1')
    md2 = Markdown('2')
    tmpl = EditableTemplate(layout={id(md2): {'width': 50, 'height': 50}})
    tmpl.main[:] = [md1, md2]
    serve_component(page, tmpl)
    items = page.locator('.muuri-grid-item')
    md1_bbox = items.nth(0).bounding_box()
    md2_bbox = items.nth(1).bounding_box()
    assert md2_bbox['y'] < md1_bbox['y']