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
def test_app_using_ipv6_and_ssl(self):
    greetings = 'Hello, World!!!'

    @webob.dec.wsgify
    def hello_world(req):
        return greetings
    server = wsgi.Server(self.conf, 'fake_ssl', hello_world, host='::1', port=0, use_ssl=True)
    server.start()
    response = requesting(method='GET', host='::1', port=server.port, ca_certs=os.path.join(SSL_CERT_DIR, 'ca.crt'), address_familly=socket.AF_INET6)
    self.assertEqual(greetings, response[-15:])
    server.stop()
    server.wait()