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
@mock.patch.object(logging.config, 'fileConfig')
def test_log_config_append(self, mock_fileConfig):
    logini = self.create_tempfiles([('log.ini', MIN_LOG_INI)])[0]
    paths = self.setup_confs('[DEFAULT]\nlog_config_append = no_exist\n', '[DEFAULT]\nlog_config_append = %s\n' % logini)
    self.assertRaises(log.LogConfigError, log.setup, self.CONF, '')
    self.assertFalse(mock_fileConfig.called)
    shutil.copy(paths[1], paths[0])
    self.CONF.mutate_config_files()
    mock_fileConfig.assert_called_once_with(logini, disable_existing_loggers=False)