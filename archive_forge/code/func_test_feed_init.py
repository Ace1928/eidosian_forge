from panel import Feed
def test_feed_init(document, comm):
    feed = Feed()
    assert feed.height == 300
    assert feed.scroll