import signal
import time
import testtools
from testtools.testcase import (
from testtools.matchers import raises
import fixtures
def sample_long_delay_with_harsh_timeout():
    with fixtures.Timeout(1, gentle=False):
        time.sleep(100)