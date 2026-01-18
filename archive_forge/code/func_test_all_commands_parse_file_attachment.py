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
def test_all_commands_parse_file_attachment(self):
    with NamedTemporaryFile() as tmp_file:
        args = safe_parse_arguments(args=[self.option, 'foo', '--attach-file', tmp_file.name])
        self.assertThat(args.attach_file.name, Equals(tmp_file.name))