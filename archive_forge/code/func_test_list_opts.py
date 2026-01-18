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
def test_list_opts(self):
    all_options = _options.list_opts()
    group, options = all_options[0]
    self.assertIsNone(group)
    self.assertEqual(_options.common_cli_opts + _options.logging_cli_opts + _options.generic_log_opts + _options.log_opts + _options.versionutils.deprecated_opts, options)