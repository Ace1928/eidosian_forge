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
def test_failed_ack_resend(self):
    """
        Ensures when a message from the manager is dropped, it gets retried until it
        works as long as there hasn't been a disconnect.
        """
    acked_packet = None
    incoming_hb = None
    message_packet = None
    hb_count = 0

    def on_ack(*args):
        nonlocal acked_packet
        acked_packet = args[0]

    def on_hb(*args):
        nonlocal incoming_hb, hb_count
        incoming_hb = args[0]
        hb_count += 1

    def on_msg(*args):
        nonlocal message_packet
        message_packet = args[0]
    self.agent1.register_to_socket(self.fake_socket, on_ack, on_hb, on_msg)
    self.assertIsNone(acked_packet)
    self.assertIsNone(incoming_hb)
    self.assertIsNone(message_packet)
    self.assertEqual(hb_count, 0)
    alive_id = self.agent1.send_alive()
    self.assertEqualBy(lambda: acked_packet is None, False, 8)
    self.assertIsNone(incoming_hb)
    self.assertIsNone(message_packet)
    self.assertIsNone(self.message_packet)
    self.assertEqualBy(lambda: self.alive_packet is None, False, 8)
    self.assertEqual(self.alive_packet.id, alive_id)
    self.assertEqual(acked_packet.id, alive_id, 'Alive was not acked')
    acked_packet = None
    self.agent1.send_heartbeat()
    self.assertEqualBy(lambda: incoming_hb is None, False, 8)
    self.assertIsNone(acked_packet)
    self.assertGreater(hb_count, 0)
    manager_message_id = 'message_id_from_manager'
    test_message_text_2 = 'test_message_text_2'
    self.agent1.send_acks = False
    message_send_packet = Packet(manager_message_id, Packet.TYPE_MESSAGE, self.socket_manager.get_my_sender_id(), TEST_WORKER_ID_1, TEST_ASSIGNMENT_ID_1, test_message_text_2, 't2')
    self.socket_manager.queue_packet(message_send_packet)
    self.assertEqualBy(lambda: message_packet is None, False, 8)
    self.assertEqual(message_packet.id, manager_message_id)
    self.assertEqual(message_packet.data, test_message_text_2)
    self.assertIn(manager_message_id, self.socket_manager.packet_map)
    self.assertNotEqual(self.socket_manager.packet_map[manager_message_id].status, Packet.STATUS_ACK)
    message_packet = None
    self.agent1.send_acks = True
    self.assertEqualBy(lambda: message_packet is None, False, 8)
    self.assertEqual(message_packet.id, manager_message_id)
    self.assertEqual(message_packet.data, test_message_text_2)
    self.assertIn(manager_message_id, self.socket_manager.packet_map)
    self.assertEqualBy(lambda: self.socket_manager.packet_map[manager_message_id].status, Packet.STATUS_ACK, 6)