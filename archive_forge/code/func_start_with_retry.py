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
def start_with_retry(self, server, port_name, max_retries, expect_launch=True, **kwargs):
    """
        Starts a server, with retries if the server launches but
        fails to start listening on the expected port.

        :param server: the server to launch
        :param port_name: the name of the port attribute
        :param max_retries: the maximum number of attempts
        :param expect_launch: true iff the server is expected to
                              successfully start
        :param expect_exit: true iff the launched process is expected
                            to exit in a timely fashion
        """
    launch_msg = None
    for i in range(max_retries):
        exitcode, out, err = server.start(expect_exit=not expect_launch, **kwargs)
        name = server.server_name
        self.assertEqual(0, exitcode, 'Failed to spin up the %s server. Got: %s' % (name, err))
        launch_msg = self.wait_for_servers([server], expect_launch)
        if launch_msg:
            server.stop()
            server.bind_port = get_unused_port()
            setattr(self, port_name, server.bind_port)
        else:
            self.launched_servers.append(server)
            break
    self.assertTrue(launch_msg is None, launch_msg)