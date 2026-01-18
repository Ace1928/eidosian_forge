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
def test_exception_with_exc_info_passed(self):
    ctxt = _fake_context()
    ctxt.request_id = str('99')
    try:
        raise RuntimeError('test_exception\ntraceback\nfrom\nremote error')
    except RuntimeError:
        self.log.exception('testing', context=ctxt)
    data = jsonutils.loads(self.stream.getvalue())
    self.assertIn('error_summary', data)
    self.assertEqual('RuntimeError: test_exception\ntraceback\nfrom\nremote error', data['error_summary'])