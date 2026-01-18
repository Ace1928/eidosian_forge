import pytest
from playwright.sync_api import expect
from panel import Feed
from panel.tests.util import serve_component, wait_until
def test_feed_view_scroll_button(page):
    feed = Feed(*list(range(1000)), height=250, scroll_button_threshold=50)
    serve_component(page, feed)
    feed_el = page.locator('.bk-panel-models-feed-Feed')
    scroll_arrow = page.locator('.scroll-button')
    expect(scroll_arrow).to_have_class('scroll-button visible')
    expect(scroll_arrow).to_be_visible()
    scroll_arrow.click()
    wait_until(lambda: feed_el.evaluate('(el) => el.scrollTop') > 0, page)
    wait_until(lambda: int(page.locator('pre').last.inner_text()) > 50, page)