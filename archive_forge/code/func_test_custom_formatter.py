import io
import logging
import time
import testtools
from testtools import TestCase
from fixtures import (
def test_custom_formatter(self):
    fixture = FakeLogger(format='%(asctime)s %(module)s', formatter=FooFormatter, datefmt='%Y')
    self.useFixture(fixture)
    logging.info('message')
    self.assertEqual(time.strftime('Foo %Y test_logger\n', time.localtime()), fixture.output)