import os
import signal
from testtools.helpers import try_import
from testtools import skipIf
from testtools.matchers import (
from ._helpers import NeedsTwistedTestCase
def test_keyword_arguments(self):
    calls = []

    def function(*a, **kw):
        return calls.extend([a, kw])
    self.make_spinner().run(self.make_timeout(), function, foo=42)
    self.assertThat(calls, Equals([(), {'foo': 42}]))