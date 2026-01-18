import io
import logging
import time
import testtools
from testtools import TestCase
from fixtures import (
def test_custom_datefmt(self):
    fixture = FakeLogger(format='%(asctime)s %(module)s', datefmt='%Y')
    self.useFixture(fixture)
    logging.info('message')
    self.assertEqual(time.strftime('%Y test_logger\n', time.localtime()), fixture.output)