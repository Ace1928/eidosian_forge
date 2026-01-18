import datetime
import optparse
from contextlib import contextmanager
from functools import partial
from io import BytesIO, TextIOWrapper
from tempfile import NamedTemporaryFile
from iso8601 import UTC
from testtools import TestCase
from testtools.matchers import (Equals, Matcher, MatchesAny, MatchesListwise,
from testtools.testresult.doubles import StreamResult
import subunit._output as _o
from subunit._output import (_ALL_ACTIONS, _FINAL_ACTIONS,
def test_all_commands_generate_tags(self):
    result = get_result_for([self.option, self.test_id, '--tag', 'hello', '--tag', 'world'])
    self.assertThat(result._events[1], MatchesStatusCall(test_tags={'hello', 'world'}))