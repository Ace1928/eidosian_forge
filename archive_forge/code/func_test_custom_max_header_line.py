import os
import platform
import socket
import tempfile
import testtools
from unittest import mock
import eventlet
import eventlet.wsgi
import requests
import webob
from oslo_config import cfg
from oslo_service import sslutils
from oslo_service.tests import base
from oslo_service import wsgi
from oslo_utils import netutils
def test_custom_max_header_line(self):
    self.config(max_header_line=4096)
    wsgi.Server(self.conf, 'test_custom_max_header_line', None)
    self.assertEqual(eventlet.wsgi.MAX_HEADER_LINE, self.conf.max_header_line)