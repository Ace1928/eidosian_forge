import unittest
import os
from unittest import mock
from parlai.mturk.core.dev.worker_manager import WorkerManager, WorkerState
from parlai.mturk.core.dev.agents import MTurkAgent, AssignState
from parlai.mturk.core.dev.mturk_manager import MTurkManager
from parlai.core.params import ParlaiParser
import parlai.mturk.core.dev.worker_manager as WorkerManagerFile
import parlai.mturk.core.dev.data_model as data_model
def test_conversation_management(self):
    """
        Tests handling conversation state, moving agents to the correct conversations,
        and disconnecting one worker in an active convo.
        """
    self.worker_manager.assign_task_to_worker(TEST_HIT_ID_1, TEST_ASSIGNMENT_ID_1, TEST_WORKER_ID_1)
    self.worker_manager.assign_task_to_worker(TEST_HIT_ID_2, TEST_ASSIGNMENT_ID_2, TEST_WORKER_ID_2)
    good_agent = self.worker_manager.get_agent_for_assignment(TEST_ASSIGNMENT_ID_1)
    bad_agent = self.worker_manager.get_agent_for_assignment(TEST_ASSIGNMENT_ID_2)

    def fake_command_send(worker_id, assignment_id, data, ack_func):
        pkt = mock.MagicMock()
        pkt.sender_id = worker_id
        pkt.assignment_id = assignment_id
        self.assertEqual(data['text'], data_model.COMMAND_CHANGE_CONVERSATION)
        ack_func(pkt)
    self.mturk_manager.send_command = fake_command_send
    good_agent.set_status(AssignState.STATUS_IN_TASK, conversation_id='t1', agent_id='good')
    bad_agent.set_status(AssignState.STATUS_IN_TASK, conversation_id='t1', agent_id='bad')
    self.assertEqual(good_agent.id, 'good')
    self.assertEqual(bad_agent.id, 'bad')
    self.assertEqual(good_agent.conversation_id, 't1')
    self.assertEqual(bad_agent.conversation_id, 't1')
    self.assertIn('t1', self.worker_manager.conv_to_agent)
    self.assertEqual(len(self.worker_manager.conv_to_agent['t1']), 2)
    self.worker_manager.handle_bad_disconnect = mock.MagicMock()
    checked_worker = False

    def partner_callback(agent):
        nonlocal checked_worker
        checked_worker = True
        self.assertEqual(agent.worker_id, good_agent.worker_id)
    self.worker_manager.handle_agent_disconnect(bad_agent.worker_id, bad_agent.assignment_id, partner_callback)
    self.assertTrue(checked_worker)
    self.worker_manager.handle_bad_disconnect.assert_called_once_with(bad_agent.worker_id)
    self.assertEqual(bad_agent.get_status(), AssignState.STATUS_DISCONNECT)