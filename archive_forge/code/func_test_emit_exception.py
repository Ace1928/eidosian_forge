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
def test_emit_exception(self):
    logger = log.getLogger('nova-exception.foo')
    local_context = _fake_context()
    try:
        raise Exception('Some exception')
    except Exception:
        logger.exception('Foo', context=local_context)
    self.assertEqual(mock.call(mock.ANY, CODE_FILE=mock.ANY, CODE_FUNC='test_emit_exception', CODE_LINE=mock.ANY, LOGGER_LEVEL='ERROR', LOGGER_NAME='nova-exception.foo', PRIORITY=3, SYSLOG_FACILITY=syslog.LOG_USER, SYSLOG_IDENTIFIER=mock.ANY, REQUEST_ID=mock.ANY, EXCEPTION_INFO=mock.ANY, EXCEPTION_TEXT=mock.ANY, PROJECT_ID='mytenant', PROJECT_NAME='mytenant', PROCESS_NAME='MainProcess', THREAD_NAME='MainThread', USER_NAME='myuser'), self.journal.send.call_args)
    args, kwargs = self.journal.send.call_args
    self.assertEqual(len(args), 1)
    self.assertIsInstance(args[0], str)
    self.assertIsInstance(kwargs['CODE_LINE'], int)
    self.assertIsInstance(kwargs['PRIORITY'], int)
    self.assertIsInstance(kwargs['SYSLOG_FACILITY'], int)
    del kwargs['CODE_LINE'], kwargs['PRIORITY'], kwargs['SYSLOG_FACILITY']
    for key, arg in kwargs.items():
        self.assertIsInstance(key, str)
        self.assertIsInstance(arg, (bytes, str))