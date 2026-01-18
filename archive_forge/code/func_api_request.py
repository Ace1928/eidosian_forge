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
def api_request(self, method, url, headers=None, data=None, json=None, body_file=None):
    """Perform a request against the API.

        NOTE: Most code should use api_get(), api_post(), api_put(),
              or api_delete() instead!

        :param method: The HTTP method to use (i.e. GET, POST, etc)
        :param url: The *path* part of the URL to call (i.e. /v2/images)
        :param headers: Optional updates to the default set of headers
        :param data: Optional bytes data payload to send (overrides @json)
        :param json: Optional dict structure to be jsonified and sent as
                     the payload (mutually exclusive with @data)
        :param body_file: Optional io.IOBase to provide as the input data
                          stream for the request (overrides @data)
        :returns: A webob.Response object
        """
    headers = self._headers(headers)
    req = webob.Request.blank(url, method=method, headers=headers)
    if json and (not data):
        data = jsonutils.dumps(json).encode()
    if data and (not body_file):
        req.body = data
    elif body_file:
        req.body_file = body_file
    return self.api(req)