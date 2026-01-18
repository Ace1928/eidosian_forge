import errno
import os
import socket
from unittest import mock
import eventlet
from oslo_service import eventlet_backdoor
from oslo_service.tests import base
@mock.patch.object(eventlet, 'spawn')
@mock.patch.object(eventlet, 'listen')
def test_backdoor_port_range_all_inuse(self, listen_mock, spawn_mock):
    self.config(backdoor_port='8800:8899')
    side_effects = []
    for i in range(8800, 8900):
        side_effects.append(socket.error(errno.EADDRINUSE, ''))
    listen_mock.side_effect = side_effects
    self.assertRaises(socket.error, eventlet_backdoor.initialize_if_enabled, self.conf)