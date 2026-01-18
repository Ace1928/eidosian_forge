import unittest
import os
import time
import uuid
from datetime import datetime
from parlai.mturk.core.dev.mturk_data_handler import MTurkDataHandler
from parlai.mturk.core.dev.agents import AssignState
import parlai.mturk.core.dev.mturk_data_handler as DataHandlerFile
def test_create_get_run(self):
    run_id = 'Test_run_1'
    hits_created = 10
    db_logger = MTurkDataHandler('test2', file_name=self.DB_NAME)
    db_logger.log_new_run(hits_created, 'testing', run_id)
    run_data = db_logger.get_run_data(run_id)
    self.assertEqual(run_data['run_id'], run_id)
    self.assertEqual(run_data['created'], 0)
    self.assertEqual(run_data['completed'], 0)
    self.assertEqual(run_data['maximum'], hits_created)
    self.assertEqual(run_data['failed'], 0)
    self.assertIsNone(db_logger.get_run_data('fake_id'))