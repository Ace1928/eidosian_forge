import unittest
import time
import uuid
import os
from unittest import mock
from parlai.mturk.core.dev.socket_manager import Packet, SocketManager
from parlai.mturk.core.dev.worlds import MTurkOnboardWorld, MTurkTaskWorld
from parlai.mturk.core.dev.agents import AssignState
from parlai.mturk.core.dev.mturk_manager import MTurkManager
from parlai.core.params import ParlaiParser
import parlai.mturk.core.dev.mturk_manager as MTurkManagerFile
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
import threading
from websocket_server import WebsocketServer
import json
def test_no_onboard_expire_waiting(self):
    manager = self.mturk_manager
    manager.set_get_onboard_world(None)
    agent_1 = self.agent_1
    self.alive_agent(agent_1)
    agent_1_object = manager.worker_manager.get_agent_for_assignment(agent_1.assignment_id)
    assert_equal_by(agent_1_object.get_status, AssignState.STATUS_WAITING, 2)
    manager._expire_agent_pool()
    self.assertEqual(agent_1_object.get_status(), AssignState.STATUS_EXPIRED)