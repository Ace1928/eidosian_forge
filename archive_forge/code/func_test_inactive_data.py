import unittest
import os
import time
import threading
from unittest import mock
from parlai.mturk.core.agents import MTurkAgent
from parlai.mturk.core.shared_utils import AssignState
from parlai.mturk.core.mturk_manager import MTurkManager
from parlai.core.params import ParlaiParser
import parlai.mturk.core.worker_manager as WorkerManagerFile
import parlai.mturk.core.data_model as data_model
def test_inactive_data(self):
    """
        Ensure data packet generated for inactive commands is valid.
        """
    for status in complete_statuses:
        self.turk_agent.set_status(status)
        data = self.turk_agent.get_inactive_command_data()
        self.assertIsNotNone(data['text'])
        self.assertIsNotNone(data['inactive_text'])
        self.assertEqual(data['conversation_id'], self.turk_agent.conversation_id)
        self.assertEqual(data['agent_id'], TEST_WORKER_ID_1)