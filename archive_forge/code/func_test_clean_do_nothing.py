import os
import signal
from testtools.helpers import try_import
from testtools import skipIf
from testtools.matchers import (
from ._helpers import NeedsTwistedTestCase
def test_clean_do_nothing(self):
    spinner = self.make_spinner()
    result = spinner._clean()
    self.assertThat(result, Equals([]))