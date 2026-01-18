import datetime
import gettext
import http.client as http
import os
import socket
from unittest import mock
import eventlet.patcher
import fixtures
from oslo_concurrency import processutils
from oslo_serialization import jsonutils
import routes
import webob
from glance.api.v2 import router as router_v2
from glance.common import exception
from glance.common import utils
from glance.common import wsgi
from glance import i18n
from glance.image_cache import prefetcher
from glance.tests import utils as test_utils
@mock.patch.object(prefetcher, 'Prefetcher')
def test_correct_configure_socket(self, mock_prefetcher):
    mock_socket = mock.Mock()
    self.useFixture(fixtures.MonkeyPatch('glance.common.wsgi.eventlet.listen', lambda *x, **y: mock_socket))
    server = wsgi.Server()
    server.default_port = 1234
    server.configure_socket()
    self.assertIn(mock.call.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1), mock_socket.mock_calls)
    self.assertIn(mock.call.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1), mock_socket.mock_calls)
    if hasattr(socket, 'TCP_KEEPIDLE'):
        self.assertIn(mock.call.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, wsgi.CONF.tcp_keepidle), mock_socket.mock_calls)