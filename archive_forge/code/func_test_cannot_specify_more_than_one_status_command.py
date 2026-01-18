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
def test_cannot_specify_more_than_one_status_command(self):

    def fn():
        return safe_parse_arguments(['--fail', 'foo', '--skip', 'bar'])
    self.assertThat(fn, raises(RuntimeError('argument --skip: Only one status may be specified at once.')))