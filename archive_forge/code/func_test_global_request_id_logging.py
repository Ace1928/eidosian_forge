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
def test_global_request_id_logging(self):
    fmt_str = 'HAS CONTEXT [%(request_id)s %(global_request_id)s]: %(message)s'
    self.config(logging_context_format_string=fmt_str)
    ctxt = _fake_context()
    ctxt.request_id = '99'
    message = 'test'
    self.log.info(message, context=ctxt)
    expected = 'HAS CONTEXT [%s %s]: %s\n' % (ctxt.request_id, ctxt.global_request_id, str(message))
    self.assertEqual(expected, self.stream.getvalue())