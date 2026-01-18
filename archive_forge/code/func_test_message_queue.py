import unittest
import os
import time
from unittest import mock
from parlai.mturk.core.dev.agents import (
from parlai.mturk.core.dev.mturk_manager import MTurkManager
from parlai.core.params import ParlaiParser
import parlai.mturk.core.dev.worker_manager as WorkerManagerFile
import parlai.mturk.core.dev.data_model as data_model
def test_message_queue(self):
    """
        Ensure observations and acts work as expected.
        """
    self.turk_agent.observe(ACT_1)
    self.mturk_manager.send_message.assert_called_with(TEST_WORKER_ID_1, TEST_ASSIGNMENT_ID_1, ACT_1)
    self.assertTrue(self.turk_agent.msg_queue.empty())
    self.turk_agent.id = AGENT_ID
    self.turk_agent.put_data(MESSAGE_ID_1, ACT_1)
    self.assertTrue(self.turk_agent.recieved_packets[MESSAGE_ID_1])
    self.assertFalse(self.turk_agent.msg_queue.empty())
    returned_act = self.turk_agent.get_new_act_message()
    self.assertEqual(returned_act, ACT_1)
    self.turk_agent.put_data(MESSAGE_ID_1, ACT_1)
    self.assertTrue(self.turk_agent.msg_queue.empty())
    for i in range(100):
        self.turk_agent.put_data(str(i), ACT_1)
    self.assertEqual(self.turk_agent.msg_queue.qsize(), 100)
    self.turk_agent.flush_msg_queue()
    self.assertTrue(self.turk_agent.msg_queue.empty())
    blank_message = self.turk_agent.get_new_act_message()
    self.assertIsNone(blank_message)
    self.turk_agent.disconnected = True
    with self.assertRaises(AgentDisconnectedError):
        self.turk_agent.get_new_act_message()
    self.turk_agent.disconnected = False
    self.turk_agent.hit_is_returned = True
    with self.assertRaises(AgentReturnedError):
        self.turk_agent.get_new_act_message()
    self.turk_agent.hit_is_returned = False
    self.turk_agent.reduce_state()
    self.assertIsNone(self.turk_agent.msg_queue)
    self.assertIsNone(self.turk_agent.recieved_packets)