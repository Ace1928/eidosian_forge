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
def test_file_name_is_used_by_default(self):
    with temp_file_contents(b'Hello') as f:
        result = get_result_for(['--attach-file', f.name])
        self.assertThat(result._events, MatchesListwise([MatchesStatusCall(call='startTestRun'), MatchesStatusCall(file_name=f.name, file_bytes=b'Hello', eof=True), MatchesStatusCall(call='stopTestRun')]))