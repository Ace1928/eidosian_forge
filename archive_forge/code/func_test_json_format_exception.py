import logging
import sys
from unittest import mock
from oslo_config import cfg
from oslo_config import fixture as config_fixture
from oslo_context import context
from oslotest import base as test_base
from oslo_log import formatters
from oslo_log import log
def test_json_format_exception(self):
    exc_info = self._unhashable_exception_info()
    formatter = formatters.JSONFormatter()
    tb = ''.join(formatter.formatException(exc_info))
    self.assertTrue(tb)