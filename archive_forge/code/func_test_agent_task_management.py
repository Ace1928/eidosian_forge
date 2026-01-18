import unittest
import os
from unittest import mock
from parlai.mturk.core.dev.worker_manager import WorkerManager, WorkerState
from parlai.mturk.core.dev.agents import MTurkAgent, AssignState
from parlai.mturk.core.dev.mturk_manager import MTurkManager
from parlai.core.params import ParlaiParser
import parlai.mturk.core.dev.worker_manager as WorkerManagerFile
import parlai.mturk.core.dev.data_model as data_model
def test_agent_task_management(self):
    """
        Ensure agents and tasks have proper bookkeeping.
        """
    self.worker_manager.assign_task_to_worker(TEST_HIT_ID_1, TEST_ASSIGNMENT_ID_1, TEST_WORKER_ID_1)
    self.worker_manager.assign_task_to_worker(TEST_HIT_ID_2, TEST_ASSIGNMENT_ID_2, TEST_WORKER_ID_2)
    self.worker_manager.assign_task_to_worker(TEST_HIT_ID_3, TEST_ASSIGNMENT_ID_3, TEST_WORKER_ID_1)
    self.assertTrue(self.worker_state_1.has_assignment(TEST_ASSIGNMENT_ID_1))
    self.assertTrue(self.worker_state_1.has_assignment(TEST_ASSIGNMENT_ID_3))
    self.assertTrue(self.worker_state_2.has_assignment(TEST_ASSIGNMENT_ID_2))
    assign_agent = self.worker_manager.get_agent_for_assignment(TEST_ASSIGNMENT_ID_1)
    self.assertEqual(assign_agent.worker_id, TEST_WORKER_ID_1)
    self.assertEqual(assign_agent.hit_id, TEST_HIT_ID_1)
    self.assertEqual(assign_agent.assignment_id, TEST_ASSIGNMENT_ID_1)
    no_such_agent = self.worker_manager.get_agent_for_assignment(FAKE_ID)
    self.assertIsNone(no_such_agent)
    checked_count = 0
    filtered_count = 0

    def check_is_worker_1(agent):
        nonlocal checked_count
        checked_count += 1
        self.assertEqual(agent.worker_id, TEST_WORKER_ID_1)

    def is_worker_1(agent):
        nonlocal filtered_count
        filtered_count += 1
        return agent.worker_id == TEST_WORKER_ID_1
    self.worker_manager.map_over_agents(check_is_worker_1, is_worker_1)
    self.assertEqual(checked_count, 2)
    self.assertEqual(filtered_count, 3)
    self.assertEqual(self.worker_manager._get_worker(TEST_WORKER_ID_1), self.worker_state_1)
    self.assertEqual(self.worker_manager._get_worker(TEST_WORKER_ID_2), self.worker_state_2)
    self.assertEqual(self.worker_manager._get_worker(TEST_WORKER_ID_3), self.worker_state_3)
    self.assertIsNone(self.worker_manager._get_worker(FAKE_ID))
    self.assertEqual(self.worker_manager._get_agent(TEST_WORKER_ID_1, TEST_ASSIGNMENT_ID_1), self.worker_manager.get_agent_for_assignment(TEST_ASSIGNMENT_ID_1))
    self.assertNotEqual(self.worker_manager._get_agent(TEST_WORKER_ID_1, TEST_ASSIGNMENT_ID_2), self.worker_manager.get_agent_for_assignment(TEST_ASSIGNMENT_ID_2))