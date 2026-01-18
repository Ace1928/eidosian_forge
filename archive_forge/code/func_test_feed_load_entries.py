import pytest
from playwright.sync_api import expect
from panel import Feed
from panel.tests.util import serve_component, wait_until
def test_feed_load_entries(page):
    feed = Feed(*list(range(1000)), height=250)
    serve_component(page, feed)
    feed_el = page.locator('.bk-panel-models-feed-Feed')
    bbox = feed_el.bounding_box()
    assert bbox['height'] == 250
    expect(feed_el).to_have_class('bk-panel-models-feed-Feed scroll-vertical')
    children_count = feed_el.locator('.bk-panel-models-markup-HTML').count()
    assert 50 <= children_count <= 65
    feed_el.evaluate('(el) => el.scrollTo({top: 100})')
    children_count = feed_el.locator('.bk-panel-models-markup-HTML').count()
    assert 50 <= children_count <= 65
    feed_el.evaluate('(el) => el.scrollTo({top: 0})')
    wait_until(lambda: feed_el.locator('.bk-panel-models-markup-HTML').count() >= 50, page)