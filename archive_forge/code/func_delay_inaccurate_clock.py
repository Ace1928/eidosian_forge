import errno
import functools
import http.client
import http.server
import io
import os
import shlex
import shutil
import signal
import socket
import subprocess
import threading
import time
from unittest import mock
from alembic import command as alembic_command
import fixtures
from oslo_concurrency import lockutils
from oslo_config import cfg
from oslo_config import fixture as cfg_fixture
from oslo_log.fixture import logging_error as log_fixture
from oslo_log import log
from oslo_utils import timeutils
from oslo_utils import units
import testtools
import webob
from glance.api.v2 import cached_images
from glance.common import config
from glance.common import exception
from glance.common import property_utils
from glance.common import utils
from glance.common import wsgi
from glance import context
from glance.db.sqlalchemy import alembic_migrations
from glance.db.sqlalchemy import api as db_api
from glance.tests.unit import fixtures as glance_fixtures
def delay_inaccurate_clock(self, duration=0.001):
    """Add a small delay to compensate for inaccurate system clocks.

        Some tests make assertions based on timestamps (e.g. comparing
        'created_at' and 'updated_at' fields). In some cases, subsequent
        time.time() calls may return identical values (python timestamps can
        have a lower resolution on Windows compared to Linux - 1e-7 as
        opposed to 1e-9).

        A small delay (a few ms should be negligeable) can prevent such
        issues. At the same time, it spares us from mocking the time
        module, which might be undesired.
        """
    if os.name == 'nt':
        time.sleep(duration)