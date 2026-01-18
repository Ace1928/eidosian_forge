import unittest
import time
import uuid
from unittest import mock
from parlai.mturk.core.socket_manager import Packet, SocketManager
from parlai.mturk.core.shared_utils import AssignState
import parlai.mturk.core.data_model as data_model
import parlai.mturk.core.shared_utils as shared_utils
import threading
from websocket_server import WebsocketServer
import json
def test_non_ack_packet_send(self):
    """
        Checks to see if non-ack'ed packets are working.
        """
    self.socket_manager._safe_send = mock.MagicMock()
    self.socket_manager._safe_put = mock.MagicMock()
    self.sent = False
    send_time = time.time()
    self.assertEqual(self.MESSAGE_SEND_PACKET_2.status, Packet.STATUS_INIT)
    self._send_packet_in_background(self.MESSAGE_SEND_PACKET_2, send_time)
    self.assertEqual(self.MESSAGE_SEND_PACKET_2.status, Packet.STATUS_SENT)
    self.socket_manager._safe_send.assert_called_once()
    self.socket_manager._safe_put.assert_not_called()
    self.assertTrue(self.sent)
    used_packet_json = self.socket_manager._safe_send.call_args[0][0]
    used_packet_dict = json.loads(used_packet_json)
    self.assertEqual(used_packet_dict['type'], data_model.SOCKET_ROUTE_PACKET_STRING)
    self.assertDictEqual(used_packet_dict['content'], self.MESSAGE_SEND_PACKET_2.as_dict())