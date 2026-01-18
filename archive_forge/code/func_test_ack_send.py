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
def test_ack_send(self):
    """
        Ensure acks are being properly created and sent.
        """
    self.socket_manager._safe_send = mock.MagicMock()
    self.socket_manager._send_ack(self.AGENT_ALIVE_PACKET)
    used_packet_json = self.socket_manager._safe_send.call_args[0][0]
    used_packet_dict = json.loads(used_packet_json)
    self.assertEqual(used_packet_dict['type'], data_model.SOCKET_ROUTE_PACKET_STRING)
    used_packet = Packet.from_dict(used_packet_dict['content'])
    self.assertEqual(self.AGENT_ALIVE_PACKET.id, used_packet.id)
    self.assertEqual(used_packet.type, Packet.TYPE_ACK)
    self.assertEqual(used_packet.sender_id, self.WORLD_ID)
    self.assertEqual(used_packet.receiver_id, self.SENDER_ID)
    self.assertEqual(used_packet.assignment_id, self.ASSIGNMENT_ID)
    self.assertEqual(used_packet.conversation_id, self.CONVERSATION_ID)
    self.assertEqual(used_packet.requires_ack, False)
    self.assertEqual(used_packet.blocking, False)
    self.assertEqual(self.AGENT_ALIVE_PACKET.status, Packet.STATUS_SENT)