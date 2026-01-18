import unittest
import os
import time
import json
import threading
import pickle
from unittest import mock
from parlai.mturk.core.dev.worker_manager import WorkerManager
from parlai.mturk.core.dev.agents import MTurkAgent, AssignState
from parlai.mturk.core.dev.worlds import MTurkOnboardWorld
from parlai.mturk.core.dev.mturk_manager import MTurkManager
from parlai.mturk.core.dev.socket_manager import SocketManager, Packet
from parlai.core.params import ParlaiParser
from websocket_server import WebsocketServer
import parlai.mturk.core.dev.mturk_manager as MTurkManagerFile
import parlai.mturk.core.dev.data_model as data_model
def test_new_message(self):
    """
        test on_new_message.
        """
    alive_packet = Packet('', TEST_WORKER_ID_1, '', '', '', {'worker_id': TEST_WORKER_ID_1, 'hit_id': TEST_HIT_ID_1, 'assignment_id': TEST_ASSIGNMENT_ID_1, 'conversation_id': None}, '')
    message_packet = Packet('', '', MTurkManagerFile.AMAZON_SNS_NAME, '', TEST_ASSIGNMENT_ID_1, {'text': MTurkManagerFile.SNS_ASSIGN_SUBMITTED}, '')
    manager = self.mturk_manager
    manager._handle_mturk_message = mock.MagicMock()
    manager.worker_manager.route_packet = mock.MagicMock()
    manager._on_new_message(alive_packet)
    manager._handle_mturk_message.assert_not_called()
    manager.worker_manager.route_packet.assert_called_once_with(alive_packet)
    manager.worker_manager.route_packet.reset_mock()
    manager._on_new_message(message_packet)
    manager._handle_mturk_message.assert_called_once_with(message_packet)
    manager.worker_manager.route_packet.assert_not_called()