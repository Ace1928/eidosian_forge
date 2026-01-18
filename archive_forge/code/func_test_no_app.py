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
def test_no_app(self):
    server = wsgi.Server(self.conf, 'test_app', None)
    self.assertEqual('test_app', server.name)