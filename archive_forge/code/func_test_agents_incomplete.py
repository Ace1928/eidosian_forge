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
def test_agents_incomplete(self):
    agents = [self.agent_1, self.agent_2, self.agent_3]
    manager = self.mturk_manager
    manager.send_state_change = mock.MagicMock()
    self.assertFalse(manager._no_agents_incomplete(agents))
    self.agent_1.set_status(AssignState.STATUS_DISCONNECT)
    self.assertFalse(manager._no_agents_incomplete(agents))
    self.agent_2.set_status(AssignState.STATUS_DONE)
    self.assertFalse(manager._no_agents_incomplete(agents))
    self.agent_3.set_status(AssignState.STATUS_PARTNER_DISCONNECT)
    self.assertFalse(manager._no_agents_incomplete(agents))
    self.agent_1.set_status(AssignState.STATUS_DONE)
    self.assertFalse(manager._no_agents_incomplete(agents))
    self.agent_3.set_status(AssignState.STATUS_DONE)
    self.assertTrue(manager._no_agents_incomplete(agents))