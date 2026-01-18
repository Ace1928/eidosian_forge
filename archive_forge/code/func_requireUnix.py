import signal
import time
import testtools
from testtools.testcase import (
from testtools.matchers import raises
import fixtures
def requireUnix(self):
    if getattr(signal, 'alarm', None) is None:
        raise TestSkipped('no alarm() function')