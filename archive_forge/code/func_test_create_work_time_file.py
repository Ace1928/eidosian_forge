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
def test_create_work_time_file(self):
    manager = self.mturk_manager
    manager._should_use_time_logs = mock.MagicMock(return_value=True)
    file_path = os.path.join(parent_dir, MTurkManagerFile.TIME_LOGS_FILE_NAME)
    file_lock = os.path.join(parent_dir, MTurkManagerFile.TIME_LOGS_FILE_LOCK)
    self.assertFalse(os.path.exists(file_lock))
    MTurkManagerFile.time.time = mock.MagicMock(return_value=42424242)
    manager._reset_time_logs(force=True)
    with open(file_path, 'rb+') as time_log_file:
        existing_times = pickle.load(time_log_file)
        self.assertEqual(existing_times['last_reset'], 42424242)
        self.assertEqual(len(existing_times), 1)
    MTurkManagerFile.time.time = mock.MagicMock(return_value=60 * 60 * 24 * 1000)
    manager._check_time_limit()
    manager.worker_manager.un_time_block_workers.assert_not_called()
    MTurkManagerFile.time.time = mock.MagicMock(return_value=60 * 60 * 24 * 1000 + 60 * 40)
    manager.time_limit_checked = 0
    manager._check_time_limit()
    manager.worker_manager.un_time_block_workers.assert_not_called()
    MTurkManagerFile.time.time = mock.MagicMock(return_value=60 * 60 * 24 * 1000)
    manager._check_time_limit()
    self.assertEqual(manager.time_limit_checked, 60 * 60 * 24 * 1000)