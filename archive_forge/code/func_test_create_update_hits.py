import unittest
import os
import time
import uuid
from datetime import datetime
from parlai.mturk.core.dev.mturk_data_handler import MTurkDataHandler
from parlai.mturk.core.dev.agents import AssignState
import parlai.mturk.core.dev.mturk_data_handler as DataHandlerFile
def test_create_update_hits(self):
    run_id = 'Test_run_2'
    hits_created = 10
    db_logger = MTurkDataHandler(file_name=self.DB_NAME)
    db_logger.log_new_run(hits_created, 'testing', run_id)
    HIT1 = self.create_hit()
    HIT2 = self.create_hit()
    HIT3 = self.create_hit()
    with self.assertRaises(AssertionError):
        db_logger.log_hit_status(HIT1)
    db_logger.log_hit_status(HIT1, run_id)
    db_logger.log_hit_status(HIT2, run_id)
    db_logger = MTurkDataHandler(run_id, file_name=self.DB_NAME)
    db_logger.log_hit_status(HIT3)
    run_data = db_logger.get_run_data(run_id)
    self.assertEqual(run_data['run_id'], run_id)
    self.assertEqual(run_data['created'], 3)
    self.assertEqual(run_data['completed'], 0)
    self.assertEqual(run_data['maximum'], hits_created)
    self.assertEqual(run_data['failed'], 0)
    for hit in [HIT1, HIT2, HIT3]:
        hit_db_data = db_logger.get_hit_data(hit['HIT']['HITId'])
        self.assertHITEqual(hit, hit_db_data, run_id)
    test_status = 'TEST_STATUS'
    HIT2['HIT']['HITStatus'] = test_status
    db_logger.log_hit_status(HIT2)
    run_data = db_logger.get_run_data(run_id)
    self.assertEqual(run_data['run_id'], run_id)
    self.assertEqual(run_data['created'], 3)
    self.assertEqual(run_data['completed'], 0)
    self.assertEqual(run_data['maximum'], hits_created)
    self.assertEqual(run_data['failed'], 0)
    for hit in [HIT1, HIT2, HIT3]:
        hit_db_data = db_logger.get_hit_data(hit['HIT']['HITId'])
        self.assertHITEqual(hit, hit_db_data, run_id)
    self.assertIsNone(db_logger.get_hit_data('fake_id'))