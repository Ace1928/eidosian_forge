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
def test_log_publish_errors_handlers(self):
    fake_handler = mock.MagicMock()
    with mock.patch('oslo_utils.importutils.import_object', return_value=fake_handler) as mock_import:
        log_dir = tempfile.mkdtemp()
        self.CONF(['--log-dir', log_dir])
        self.CONF.set_default('use_stderr', False)
        self.CONF.set_default('publish_errors', True)
        log._setup_logging_from_conf(self.CONF, 'test', 'test')
        logger = log._loggers[None].logger
        self.assertEqual(2, len(logger.handlers))
        self.assertIsInstance(logger.handlers[0], logging.handlers.WatchedFileHandler)
        self.assertEqual(fake_handler, logger.handlers[1])
        mock_import.assert_called_once_with('oslo_messaging.notify.log_handler.PublishErrorsHandler', logging.ERROR)