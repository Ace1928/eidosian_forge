from testtools import TestCase
from testtools.matchers import Equals, MatchesException, Raises
from testtools.content_type import (
def test___init___sets_ivars(self):
    content_type = ContentType('foo', 'bar')
    self.assertEqual('foo', content_type.type)
    self.assertEqual('bar', content_type.subtype)
    self.assertEqual({}, content_type.parameters)