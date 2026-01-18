from panel import Feed
def test_feed_set_objects(document, comm):
    feed = Feed(height=100)
    feed.objects = list(range(1000))
    assert feed.objects == list(range(1000))