import datetime
import logging
import time
from unittest import mock
import iso8601
from oslotest import base as test_base
from testtools import matchers
from oslo_utils import timeutils
@timeutils.time_it(fake_logger)
def broken_function():
    raise IOError('Broken')