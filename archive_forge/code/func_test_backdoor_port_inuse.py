import errno
import os
import socket
from unittest import mock
import eventlet
from oslo_service import eventlet_backdoor
from oslo_service.tests import base
@mock.patch.object(eventlet, 'spawn')
@mock.patch.object(eventlet, 'listen')
def test_backdoor_port_inuse(self, listen_mock, spawn_mock):
    self.config(backdoor_port=2345)
    listen_mock.side_effect = socket.error(errno.EADDRINUSE, '')
    self.assertRaises(socket.error, eventlet_backdoor.initialize_if_enabled, self.conf)