import unittest
import os
import time
from unittest import mock
from parlai.mturk.core.dev.agents import (
from parlai.mturk.core.dev.mturk_manager import MTurkManager
from parlai.core.params import ParlaiParser
import parlai.mturk.core.dev.worker_manager as WorkerManagerFile
import parlai.mturk.core.dev.data_model as data_model
def test_assign_state_init(self):
    """
        Test proper initialization of assignment states.
        """
    self.assertEqual(self.agent_state1.status, AssignState.STATUS_NONE)
    self.assertEqual(len(self.agent_state1.messages), 0)
    self.assertEqual(len(self.agent_state1.message_ids), 0)
    self.assertEqual(self.agent_state2.status, AssignState.STATUS_IN_TASK)
    self.assertEqual(len(self.agent_state1.messages), 0)
    self.assertEqual(len(self.agent_state1.message_ids), 0)