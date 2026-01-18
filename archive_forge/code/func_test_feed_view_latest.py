import pytest
from playwright.sync_api import expect
from panel import Feed
from panel.tests.util import serve_component, wait_until
def test_feed_view_latest(page):
    feed = Feed(*list(range(1000)), height=250, view_latest=True)
    serve_component(page, feed)
    feed_el = page.locator('.bk-panel-models-feed-Feed')
    bbox = feed_el.bounding_box()
    assert bbox['height'] == 250
    expect(feed_el).to_have_class('bk-panel-models-feed-Feed scroll-vertical')
    assert feed_el.evaluate('(el) => el.scrollTop') > 0
    wait_until(lambda: int(page.locator('pre').last.inner_text()) > 950, page)