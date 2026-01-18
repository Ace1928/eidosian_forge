import asyncore
import errno
import socket
import threading
from taskflow.engines.action_engine import process_executor as pu
from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils as test_utils
@mock.patch('socket.socket')
def test_no_connect_channel(self, mock_socket_factory):
    mock_sock = mock.MagicMock()
    mock_socket_factory.return_value = mock_sock
    mock_sock.connect.side_effect = socket.error(errno.ECONNREFUSED, 'broken')
    c = pu.Channel(2222, b'me', b'secret')
    self.assertRaises(socket.error, c.send, 'hi')
    self.assertTrue(c.dead)
    self.assertTrue(mock_sock.close.called)