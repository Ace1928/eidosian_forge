from testtools import TestCase
from testtools.matchers import Equals, MatchesException, Raises
from testtools.content_type import (
def test___init___None_errors(self):
    raises_value_error = Raises(MatchesException(ValueError))
    self.assertThat(lambda: ContentType(None, None), raises_value_error)
    self.assertThat(lambda: ContentType(None, 'traceback'), raises_value_error)
    self.assertThat(lambda: ContentType('text', None), raises_value_error)