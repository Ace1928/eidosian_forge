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
def test_filename_can_be_overridden(self):
    with temp_file_contents(b'Hello') as f:
        specified_file_name = self.getUniqueString()
        result = get_result_for(['--attach-file', f.name, '--file-name', specified_file_name])
        self.assertThat(result._events, MatchesListwise([MatchesStatusCall(call='startTestRun'), MatchesStatusCall(file_name=specified_file_name, file_bytes=b'Hello'), MatchesStatusCall(call='stopTestRun')]))