import errno
import os
import socket
from unittest import mock
import eventlet
from oslo_service import eventlet_backdoor
from oslo_service.tests import base
@mock.patch.object(eventlet, 'spawn')
def test_backdoor_port_range_inuse(self, spawn_mock):
    self.config(backdoor_port='8800:8801')
    port = eventlet_backdoor.initialize_if_enabled(self.conf)
    self.assertEqual(8800, port)
    port = eventlet_backdoor.initialize_if_enabled(self.conf)
    self.assertEqual(8801, port)