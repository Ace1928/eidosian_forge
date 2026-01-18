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
def test_connection_ids(self):
    """
        Ensure that connection ids are reported as we expect them.
        """
    sender_conn_id = '{}_{}'.format(self.SENDER_ID, self.ASSIGNMENT_ID)
    receiver_conn_id = '{}_{}'.format(self.RECEIVER_ID, self.ASSIGNMENT_ID)
    self.assertEqual(self.packet_1.get_sender_connection_id(), sender_conn_id)
    self.assertEqual(self.packet_1.get_receiver_connection_id(), receiver_conn_id)