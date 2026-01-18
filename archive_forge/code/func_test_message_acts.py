import unittest
import os
import time
from unittest import mock
from parlai.mturk.core.dev.agents import (
from parlai.mturk.core.dev.mturk_manager import MTurkManager
from parlai.core.params import ParlaiParser
import parlai.mturk.core.dev.worker_manager as WorkerManagerFile
import parlai.mturk.core.dev.data_model as data_model
def test_message_acts(self):
    self.mturk_manager.handle_turker_timeout = mock.MagicMock()
    self.assertIsNone(self.turk_agent.message_request_time)
    returned_act = self.turk_agent.act(blocking=False)
    self.assertIsNotNone(self.turk_agent.message_request_time)
    self.assertIsNone(returned_act)
    self.turk_agent.id = AGENT_ID
    self.turk_agent.put_data(MESSAGE_ID_1, ACT_1)
    returned_act = self.turk_agent.act(blocking=False)
    self.assertIsNone(self.turk_agent.message_request_time)
    self.assertEqual(returned_act, ACT_1)
    self.mturk_manager.send_command.assert_called_once()
    with self.assertRaises(AgentTimeoutError):
        self.mturk_manager.send_command = mock.MagicMock()
        returned_act = self.turk_agent.act(timeout=0.07, blocking=False)
        self.assertIsNotNone(self.turk_agent.message_request_time)
        self.assertIsNone(returned_act)
        while returned_act is None:
            returned_act = self.turk_agent.act(timeout=0.07, blocking=False)
    with self.assertRaises(AgentTimeoutError):
        returned_act = self.turk_agent.act(timeout=0.07)