import pytest
from playwright.sync_api import expect
from panel.layout.base import _SCROLL_MAPPING, Column
from panel.layout.spacer import Spacer
from panel.tests.util import serve_component, wait_until
def test_column_view_latest(page):
    col = Column(Spacer(styles=dict(background='red'), width=200, height=200), Spacer(styles=dict(background='green'), width=200, height=200), Spacer(styles=dict(background='blue'), width=200, height=200), view_latest=True, scroll=True, height=420)
    serve_component(page, col)
    column = page.locator('.bk-panel-models-layout-Column')
    bbox = column.bounding_box()
    assert bbox['width'] in (200, 215)
    assert bbox['height'] == 420
    expect(column).to_have_class('bk-panel-models-layout-Column scrollable-vertical')
    expect(column).not_to_have_js_property('scrollTop', '0')