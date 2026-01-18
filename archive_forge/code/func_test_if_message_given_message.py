from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._higherorder import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
def test_if_message_given_message(self):
    matcher = Equals(1)
    expected = Annotate('foo', matcher)
    annotated = Annotate.if_message('foo', matcher)
    self.assertThat(annotated, MatchesStructure.fromExample(expected, 'annotation', 'matcher'))