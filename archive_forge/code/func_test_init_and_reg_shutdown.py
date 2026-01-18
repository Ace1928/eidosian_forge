import unittest
import time
import uuid
from unittest import mock
from parlai.mturk.core.dev.socket_manager import Packet, SocketManager
from parlai.mturk.core.dev.agents import AssignState
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
import threading
from websocket_server import WebsocketServer
import json
def test_init_and_reg_shutdown(self):
    """
        Test initialization of a socket manager.
        """
    self.assertFalse(self.fake_socket.connected)
    nop_called = False

    def nop(*args):
        nonlocal nop_called
        nop_called = True
    socket_manager = SocketManager('https://127.0.0.1', self.fake_socket.port, nop, nop, nop, TASK_GROUP_ID_1, 0.3, nop)
    self.assertTrue(self.fake_socket.connected)
    self.assertFalse(nop_called)
    self.assertFalse(self.fake_socket.disconnected)
    self.assertFalse(socket_manager.is_shutdown)
    self.assertTrue(socket_manager.alive)
    socket_manager.shutdown()
    time.sleep(0.3)
    self.assertTrue(self.fake_socket.disconnected)
    self.assertTrue(socket_manager.is_shutdown)
    self.assertFalse(nop_called)