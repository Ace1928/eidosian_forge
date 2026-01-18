import unittest
import os
import time
from unittest import mock
from parlai.mturk.core.dev.agents import (
from parlai.mturk.core.dev.mturk_manager import MTurkManager
from parlai.core.params import ParlaiParser
import parlai.mturk.core.dev.worker_manager as WorkerManagerFile
import parlai.mturk.core.dev.data_model as data_model
def test_message_management(self):
    """
        Test message management in an AssignState.
        """
    self.agent_state1.append_message(MESSAGE_1)
    self.assertEqual(len(self.agent_state1.get_messages()), 1)
    self.agent_state1.append_message(MESSAGE_2)
    self.assertEqual(len(self.agent_state1.get_messages()), 2)
    self.agent_state1.append_message(MESSAGE_1)
    self.assertEqual(len(self.agent_state1.get_messages()), 2)
    self.assertEqual(len(self.agent_state2.get_messages()), 0)
    self.assertIn(MESSAGE_1, self.agent_state1.get_messages())
    self.assertIn(MESSAGE_2, self.agent_state1.get_messages())
    self.assertEqual(len(self.agent_state1.message_ids), 2)
    self.agent_state2.append_message(MESSAGE_1)
    self.assertEqual(len(self.agent_state2.message_ids), 1)
    self.agent_state1.clear_messages()
    self.assertEqual(len(self.agent_state1.messages), 0)
    self.assertEqual(len(self.agent_state1.message_ids), 0)
    self.assertEqual(len(self.agent_state2.message_ids), 1)