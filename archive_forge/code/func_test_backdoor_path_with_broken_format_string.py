import errno
import os
import socket
from unittest import mock
import eventlet
from oslo_service import eventlet_backdoor
from oslo_service.tests import base
@mock.patch.object(eventlet, 'spawn')
@mock.patch.object(eventlet, 'listen')
def test_backdoor_path_with_broken_format_string(self, listen_mock, spawn_mock):
    broken_socket_paths = ['/tmp/my_special_socket-{}', '/tmp/my_special_socket-{broken', '/tmp/my_special_socket-{broken}']
    for socket_path in broken_socket_paths:
        self.config(backdoor_socket=socket_path)
        listen_mock.side_effect = mock.Mock()
        path = eventlet_backdoor.initialize_if_enabled(self.conf)
        self.assertEqual(socket_path, path)