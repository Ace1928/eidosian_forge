import pytest
from playwright.sync_api import expect
from panel import Feed
from panel.tests.util import serve_component, wait_until
def test_feed_dynamic_objects(page):
    feed = Feed(height=250, load_buffer=10)
    serve_component(page, feed)
    feed.objects = list(range(1000))
    expect(page.locator('pre').nth(0)).to_have_text('0')
    wait_until(lambda: page.locator('pre').count() > 10, page)