import datetime
import logging
import time
from unittest import mock
import iso8601
from oslotest import base as test_base
from testtools import matchers
from oslo_utils import timeutils
def monotonic_iter(start=0, incr=0.05):
    while True:
        yield start
        start += incr