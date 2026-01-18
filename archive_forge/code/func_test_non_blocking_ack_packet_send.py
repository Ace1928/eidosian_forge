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
def test_non_blocking_ack_packet_send(self):
    """
        Checks to see if ack'ed non-blocking packets are working.
        """
    self.socket_manager._safe_send = mock.MagicMock()
    self.socket_manager._safe_put = mock.MagicMock()
    self.sent = False
    send_time = time.time()
    self.assertEqual(self.MESSAGE_SEND_PACKET_3.status, Packet.STATUS_INIT)
    self._send_packet_in_background(self.MESSAGE_SEND_PACKET_3, send_time)
    self.assertEqual(self.MESSAGE_SEND_PACKET_3.status, Packet.STATUS_SENT)
    self.socket_manager._safe_send.assert_called_once()
    self.socket_manager._safe_put.assert_called_once()
    self.assertTrue(self.sent)
    call_args = self.socket_manager._safe_put.call_args[0]
    connection_id = call_args[0]
    queue_item = call_args[1]
    self.assertEqual(connection_id, self.MESSAGE_SEND_PACKET_3.get_receiver_connection_id())
    expected_send_time = send_time + SocketManager.ACK_TIME[self.MESSAGE_SEND_PACKET_3.type]
    self.assertAlmostEqual(queue_item[0], expected_send_time, places=2)
    self.assertEqual(queue_item[1], self.MESSAGE_SEND_PACKET_3)
    used_packet_json = self.socket_manager._safe_send.call_args[0][0]
    used_packet_dict = json.loads(used_packet_json)
    self.assertEqual(used_packet_dict['type'], data_model.SOCKET_ROUTE_PACKET_STRING)
    self.assertDictEqual(used_packet_dict['content'], self.MESSAGE_SEND_PACKET_3.as_dict())