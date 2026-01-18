import io
import logging
import time
import testtools
from testtools import TestCase
from fixtures import (
def test_exceptionraised(self):
    with FakeLogger():
        with testtools.ExpectedException(TypeError):
            logging.info('Some message', 'wrongarg')