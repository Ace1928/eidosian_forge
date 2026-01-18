import errno
import os
import socket
from unittest import mock
import eventlet
from oslo_service import eventlet_backdoor
from oslo_service.tests import base
@mock.patch.object(eventlet, 'spawn')
@mock.patch.object(eventlet, 'listen')
def test_backdoor_path_with_format_string(self, listen_mock, spawn_mock):
    self.config(backdoor_socket='/tmp/my_special_socket-{pid}')
    listen_mock.side_effect = mock.Mock()
    path = eventlet_backdoor.initialize_if_enabled(self.conf)
    expected_path = '/tmp/my_special_socket-{}'.format(os.getpid())
    self.assertEqual(expected_path, path)