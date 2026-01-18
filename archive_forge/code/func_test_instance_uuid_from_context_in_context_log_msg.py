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
def test_instance_uuid_from_context_in_context_log_msg(self):
    ctxt = _fake_context()
    ctxt.instance_uuid = 'CCCCCCCC-8A12-4C53-A736-D7A1C36A62F3'
    message = 'info'
    self.log.info(message, context=ctxt)
    expected = '[instance: %s]' % ctxt.instance_uuid
    self.assertIn(expected, self.stream.getvalue())