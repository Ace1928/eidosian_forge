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
def test_can_read_stdin(self):
    self.patch(_o.sys, 'stdin', TextIOWrapper(BytesIO(b'\xfe\xed\xfa\xce')))
    result = get_result_for([self.option, self.test_id, '--attach-file', '-'])
    self.assertThat(result._events, MatchesListwise([MatchesStatusCall(call='startTestRun'), MatchesStatusCall(file_bytes=b'\xfe\xed\xfa\xce', file_name='stdin', eof=True), MatchesStatusCall(call='stopTestRun')]))