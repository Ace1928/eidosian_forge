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
def test_full_lifecycle(self):
    manager = self.mturk_manager
    server_url = 'https://fake_server_url'
    topic_arn = 'aws_topic_arn'
    mturk_page_url = 'https://test_mturk_page_url'
    MTurkManagerFile.server_utils.setup_server = mock.MagicMock(return_value=server_url)
    with self.assertRaises(AssertionError):
        manager.start_new_run()
    with self.assertRaises(AssertionError):
        manager.start_task(None, None, None)
    manager.opt['local'] = True
    manager.opt['frontend_version'] = 1
    MTurkManagerFile.input = mock.MagicMock()
    MTurkManagerFile.mturk_utils.setup_aws_credentials = mock.MagicMock()
    MTurkManagerFile.mturk_utils.check_mturk_balance = mock.MagicMock(return_value=False)
    MTurkManagerFile.mturk_utils.calculate_mturk_cost = mock.MagicMock(return_value=10)
    with self.assertRaises(SystemExit):
        manager.setup_server()
    MTurkManagerFile.mturk_utils.setup_aws_credentials.assert_called_once()
    MTurkManagerFile.mturk_utils.check_mturk_balance.assert_called_once()
    MTurkManagerFile.input.assert_called()
    self.assertEqual(len(MTurkManagerFile.input.call_args_list), 2)
    manager.opt['local'] = False
    MTurkManagerFile.input.reset_mock()
    MTurkManagerFile.mturk_utils.check_mturk_balance = mock.MagicMock(return_value=True)
    MTurkManagerFile.mturk_utils.create_hit_config = mock.MagicMock()
    manager.setup_server()
    self.assertEqual(len(manager.task_files_to_copy), 4)
    self.assertEqual(manager.server_url, server_url)
    self.assertIn('unittest', manager.server_task_name)
    MTurkManagerFile.input.assert_called_once()
    MTurkManagerFile.mturk_utils.check_mturk_balance.assert_called_once()
    MTurkManagerFile.mturk_utils.create_hit_config.assert_called_once()
    self.assertEqual(manager.task_state, manager.STATE_SERVER_ALIVE)
    MTurkManagerFile.mturk_utils.setup_sns_topic = mock.MagicMock(return_value=topic_arn)
    manager._init_state = mock.MagicMock(wraps=manager._init_state)
    manager.start_new_run()
    manager._init_state.assert_called_once()
    MTurkManagerFile.mturk_utils.setup_sns_topic.assert_called_once_with(manager.opt['task'], manager.server_url, manager.task_group_id)
    self.assertEqual(manager.topic_arn, topic_arn)
    self.assertEqual(manager.task_state, manager.STATE_INIT_RUN)
    manager._setup_socket = mock.MagicMock()
    manager.ready_to_accept_workers()
    manager._setup_socket.assert_called_once()
    self.assertEqual(manager.task_state, MTurkManager.STATE_ACCEPTING_WORKERS)
    manager.create_additional_hits = mock.MagicMock(return_value=mturk_page_url)
    hits_url = manager.create_hits()
    manager.create_additional_hits.assert_called_once()
    self.assertEqual(manager.task_state, MTurkManager.STATE_HITS_MADE)
    self.assertEqual(hits_url, mturk_page_url)
    manager.num_conversations = 10
    manager.expire_all_unassigned_hits = mock.MagicMock()
    manager._expire_onboarding_pool = mock.MagicMock()
    manager._expire_agent_pool = mock.MagicMock()

    def run_task():
        manager.start_task(lambda worker: True, None, None)
    task_thread = threading.Thread(target=run_task, daemon=True)
    task_thread.start()
    self.assertTrue(task_thread.isAlive())
    manager.started_conversations = 10
    manager.completed_conversations = 10
    assert_equal_by(task_thread.isAlive, False, 0.6)
    manager.expire_all_unassigned_hits.assert_called_once()
    manager._expire_onboarding_pool.assert_called_once()
    manager._expire_agent_pool.assert_called_once()
    manager.expire_all_hits = mock.MagicMock()
    manager._expire_onboarding_pool = mock.MagicMock()
    manager._expire_agent_pool = mock.MagicMock()
    manager._wait_for_task_expirations = mock.MagicMock()
    MTurkManagerFile.mturk_utils.delete_sns_topic = mock.MagicMock()
    manager.shutdown()
    self.assertTrue(manager.is_shutdown)
    manager.expire_all_hits.assert_called_once()
    manager._expire_onboarding_pool.assert_called_once()
    manager._expire_agent_pool.assert_called_once()
    manager._wait_for_task_expirations.assert_called_once()
    MTurkManagerFile.server_utils.delete_server.assert_called_once()
    MTurkManagerFile.mturk_utils.delete_sns_topic.assert_called_once_with(topic_arn)