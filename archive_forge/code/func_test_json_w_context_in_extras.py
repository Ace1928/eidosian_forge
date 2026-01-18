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
def test_json_w_context_in_extras(self):
    test_msg = 'This is a %(test)s line'
    test_data = {'test': 'log'}
    local_context = _fake_context()
    self.log.debug(test_msg, test_data, key='value', context=local_context)
    self._validate_json_data('test_json_w_context_in_extras', test_msg, test_data, local_context)