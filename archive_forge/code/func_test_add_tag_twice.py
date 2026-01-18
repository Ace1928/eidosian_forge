from testtools import TestCase
from testtools.tags import TagContext
def test_add_tag_twice(self):
    tag_context = TagContext()
    tag_context.change_tags({'foo'}, set())
    tag_context.change_tags({'bar'}, set())
    self.assertEqual({'foo', 'bar'}, tag_context.get_current_tags())