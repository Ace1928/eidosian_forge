from contextlib import contextmanager
import copy
import datetime
import io
import logging
import os
import platform
import shutil
import sys
import tempfile
import time
from unittest import mock
from dateutil import tz
from oslo_config import cfg
from oslo_config import fixture as fixture_config  # noqa
from oslo_context import context
from oslo_context import fixture as fixture_context
from oslo_i18n import fixture as fixture_trans
from oslo_serialization import jsonutils
from oslotest import base as test_base
import testtools
from oslo_log import _options
from oslo_log import formatters
from oslo_log import handlers
from oslo_log import log
from oslo_utils import units
def test_no_exception_logging_format_string(self):
    self.config(logging_context_format_string='%(error_summary)s')
    ctxt = _fake_context()
    ctxt.request_id = str('99')
    message = self.trans_fixture.lazy('test ' + chr(128))
    self.log.info(message, context=ctxt)
    expected = '-\n'
    self.assertTrue(self.stream.getvalue().startswith(expected))