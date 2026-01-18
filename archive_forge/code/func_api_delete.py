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
def api_delete(self, url, headers=None):
    """Perform a DELETE request against the API.

        :param url: The *path* part of the URL to call (i.e. /v2/images)
        :param headers: Optional updates to the default set of headers
        :returns: A webob.Response object
        """
    return self.api_request('DELETE', url, headers=headers)