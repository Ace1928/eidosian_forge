from testtools import TestCase
from testtools.tags import TagContext
def test_change_tags_returns_tags(self):
    tag_context = TagContext()
    tags = tag_context.change_tags({'foo'}, set())
    self.assertEqual({'foo'}, tags)