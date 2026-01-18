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
def test_break_multi_convo(self):
    manager = self.mturk_manager
    manager.opt['allowed_conversations'] = 1
    agent_1 = self.agent_1
    self.alive_agent(agent_1)
    assert_equal_by(lambda: agent_1.worker_id in self.onboarding_agents, True, 2)
    agent_1_object = manager.worker_manager.get_agent_for_assignment(agent_1.assignment_id)
    self.assertFalse(self.onboarding_agents[agent_1.worker_id])
    self.assertEqual(agent_1_object.get_status(), AssignState.STATUS_ONBOARDING)
    self.onboarding_agents[agent_1.worker_id] = True
    assert_equal_by(agent_1_object.get_status, AssignState.STATUS_WAITING, 2)
    agent_2 = self.agent_2
    self.alive_agent(agent_2)
    assert_equal_by(lambda: agent_2.worker_id in self.onboarding_agents, True, 2)
    agent_2_object = manager.worker_manager.get_agent_for_assignment(agent_2.assignment_id)
    self.assertFalse(self.onboarding_agents[agent_2.worker_id])
    self.assertEqual(agent_2_object.get_status(), AssignState.STATUS_ONBOARDING)
    self.onboarding_agents[agent_2.worker_id] = True
    assert_equal_by(agent_2_object.get_status, AssignState.STATUS_IN_TASK, 2)
    assert_equal_by(lambda: agent_2.worker_id in self.worlds_agents, True, 2)
    self.assertIn(agent_1.worker_id, self.worlds_agents)
    agent_1_2 = self.agent_1_2
    self.alive_agent(agent_1_2)
    assert_equal_by(lambda: agent_1_2.worker_id in self.onboarding_agents, True, 2)
    agent_1_2_object = manager.worker_manager.get_agent_for_assignment(agent_1_2.assignment_id)
    self.assertIsNone(agent_1_2_object)
    self.worlds_agents[agent_1.worker_id] = True
    self.worlds_agents[agent_2.worker_id] = True
    agent_1_object.set_completed_act({})
    agent_2_object.set_completed_act({})
    assert_equal_by(agent_1_object.get_status, AssignState.STATUS_DONE, 2)
    assert_equal_by(agent_2_object.get_status, AssignState.STATUS_DONE, 2)
    assert_equal_by(lambda: manager.completed_conversations, 1, 2)