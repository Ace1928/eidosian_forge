import logging
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_utils import excutils
from oslo_utils import timeutils
@excutils.forever_retry_uncaught_exceptions
def exception_generator(self):
    while self._exceptions:
        raise self._exceptions.pop(0)