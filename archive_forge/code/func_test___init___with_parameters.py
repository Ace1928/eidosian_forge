from testtools import TestCase
from testtools.matchers import Equals, MatchesException, Raises
from testtools.content_type import (
def test___init___with_parameters(self):
    content_type = ContentType('foo', 'bar', {'quux': 'thing'})
    self.assertEqual({'quux': 'thing'}, content_type.parameters)