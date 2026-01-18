import io
import warnings
import sys
from fixtures import CompoundFixture, Fixture
from testtools.content import Content, text_content
from testtools.content_type import UTF8_TEXT
from testtools.runtest import RunTest, _raise_force_fail_error
from ._deferred import extract_result
from ._spinner import (
from twisted.internet import defer
from twisted.python import log
def force_failure(ignored):
    if getattr(self.case, 'force_failure', None):
        d = self._run_user(_raise_force_fail_error)
        d.addCallback(fails.append)
        return d