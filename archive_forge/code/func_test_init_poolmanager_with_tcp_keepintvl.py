import datetime
import io
import itertools
import json
import logging
import sys
from unittest import mock
import uuid
from oslo_utils import encodeutils
import requests
import requests.auth
from testtools import matchers
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import plugin
from keystoneauth1 import session as client_session
from keystoneauth1.tests.unit import utils
from keystoneauth1 import token_endpoint
def test_init_poolmanager_with_tcp_keepintvl(self):
    self.patch(client_session, 'REQUESTS_VERSION', (2, 4, 1))
    socket = self.patch_socket_with_options(['IPPROTO_TCP', 'TCP_NODELAY', 'SOL_SOCKET', 'SO_KEEPALIVE', 'TCP_KEEPINTVL'])
    given_adapter = client_session.TCPKeepAliveAdapter()
    given_adapter.init_poolmanager(1, 2, 3)
    self.init_poolmanager.assert_called_once_with(1, 2, 3, socket_options=[(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1), (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1), (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 15)])