from testtools import TestCase
from testtools.matchers import Equals, MatchesException, Raises
from testtools.content_type import (
def test_json_content(self):
    self.assertThat(JSON.type, Equals('application'))
    self.assertThat(JSON.subtype, Equals('json'))
    self.assertThat(JSON.parameters, Equals({}))