import abc
import atexit
import datetime
import errno
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
from testtools import content as ttc
import textwrap
import time
from unittest import mock
import urllib.parse as urlparse
import uuid
import fixtures
import glance_store
from os_win import utilsfactory as os_win_utilsfactory
from oslo_config import cfg
from oslo_serialization import jsonutils
import testtools
import webob
from glance.common import config
from glance.common import utils
from glance.common import wsgi
from glance.db.sqlalchemy import api as db_api
from glance import tests as glance_tests
from glance.tests import utils as test_utils
import glance.async_
def dump_log(self):
    if not self.log_file:
        return 'log_file not set for {name}'.format(name=self.server_name)
    elif not os.path.exists(self.log_file):
        return '{log_file} for {name} did not exist'.format(log_file=self.log_file, name=self.server_name)
    with open(self.log_file, 'r') as fptr:
        return fptr.read().strip()