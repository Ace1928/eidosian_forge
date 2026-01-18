import pytest
from playwright.sync_api import expect
from panel.layout.base import _SCROLL_MAPPING, Column
from panel.layout.spacer import Spacer
from panel.tests.util import serve_component, wait_until
def test_column_scroll(page):
    col = Column(Spacer(styles=dict(background='red'), width=200, height=200), Spacer(styles=dict(background='green'), width=200, height=200), Spacer(styles=dict(background='blue'), width=200, height=200), scroll=True, height=420)
    serve_component(page, col)
    col_el = page.locator('.bk-panel-models-layout-Column')
    bbox = col_el.bounding_box()
    assert bbox['width'] in (200, 215)
    assert bbox['height'] == 420
    expect(col_el).to_have_class('bk-panel-models-layout-Column scrollable-vertical')