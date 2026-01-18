import unittest
import os
from unittest import mock
from parlai.mturk.core.dev.worker_manager import WorkerManager, WorkerState
from parlai.mturk.core.dev.agents import MTurkAgent, AssignState
from parlai.mturk.core.dev.mturk_manager import MTurkManager
from parlai.core.params import ParlaiParser
import parlai.mturk.core.dev.worker_manager as WorkerManagerFile
import parlai.mturk.core.dev.data_model as data_model
def test_disconnect_management(self):
    self.worker_manager.load_disconnects()
    self.worker_manager.is_sandbox = False
    self.mturk_manager.block_worker = mock.MagicMock()
    self.mturk_manager.soft_block_worker = mock.MagicMock()
    self.assertEqual(len(self.worker_manager.disconnects), 0)
    self.worker_manager.handle_bad_disconnect(TEST_WORKER_ID_1)
    self.assertEqual(len(self.worker_manager.disconnects), 1)
    self.mturk_manager.block_worker.assert_not_called()
    self.mturk_manager.soft_block_worker.assert_not_called()
    self.worker_manager.handle_bad_disconnect(TEST_WORKER_ID_1)
    self.assertEqual(len(self.worker_manager.disconnects), 2)
    self.mturk_manager.block_worker.assert_not_called()
    self.mturk_manager.soft_block_worker.assert_not_called()
    self.assertEqual(self.worker_manager.mturk_workers[TEST_WORKER_ID_1].disconnects, 2)
    self.worker_manager.handle_bad_disconnect(TEST_WORKER_ID_2)
    self.assertEqual(len(self.worker_manager.disconnects), 3)
    self.mturk_manager.block_worker.assert_not_called()
    self.mturk_manager.soft_block_worker.assert_not_called()
    self.worker_manager.opt['disconnect_qualification'] = 'test'
    self.worker_manager.handle_bad_disconnect(TEST_WORKER_ID_1)
    self.mturk_manager.block_worker.assert_not_called()
    self.mturk_manager.soft_block_worker.assert_called_with(TEST_WORKER_ID_1, 'disconnect_qualification')
    self.mturk_manager.soft_block_worker.reset_mock()
    self.worker_manager.opt['hard_block'] = True
    self.worker_manager.handle_bad_disconnect(TEST_WORKER_ID_2)
    self.mturk_manager.block_worker.assert_called_once()
    self.mturk_manager.soft_block_worker.assert_not_called()
    self.worker_manager.save_disconnects()
    worker_manager2 = WorkerManager(self.mturk_manager, self.opt)
    self.assertEqual(len(worker_manager2.disconnects), 5)
    self.assertEqual(worker_manager2.mturk_workers[TEST_WORKER_ID_1].disconnects, 3)
    self.assertEqual(worker_manager2.mturk_workers[TEST_WORKER_ID_2].disconnects, 2)
    worker_manager2.shutdown()