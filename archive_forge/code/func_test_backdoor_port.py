import errno
import os
import socket
from unittest import mock
import eventlet
from oslo_service import eventlet_backdoor
from oslo_service.tests import base
@mock.patch.object(eventlet, 'spawn')
@mock.patch.object(eventlet, 'listen')
def test_backdoor_port(self, listen_mock, spawn_mock):
    self.config(backdoor_port=1234)
    sock = mock.Mock()
    sock.getsockname.return_value = ('127.0.0.1', 1234)
    listen_mock.return_value = sock
    port = eventlet_backdoor.initialize_if_enabled(self.conf)
    self.assertEqual(1234, port)