import json
import logging
import os
import pickle
import sqlite3
import time
import threading
import parlai.mturk.core.dev.shared_utils as shared_utils
from parlai.mturk.core.dev.agents import AssignState
def log_complete_assignment(self, worker_id, assignment_id, approve_time, complete_type, task_group_id=None):
    """
        Note that an assignment was completed.
        """
    task_group_id = self._force_task_group_id(task_group_id)
    with self.table_access_condition:
        conn = self._get_connection()
        c = conn.cursor()
        c.execute('UPDATE assignments SET status = ?, approve_time = ?\n                         WHERE assignment_id = ?;', ('Completed', approve_time, assignment_id))
        c.execute('UPDATE workers SET completed = completed + 1\n                         WHERE worker_id = ?;', (worker_id,))
        c.execute('UPDATE pairings SET status = ?, task_end = ?\n                         WHERE worker_id = ? AND assignment_id = ?;', (complete_type, time.time(), worker_id, assignment_id))
        c.execute('UPDATE runs SET completed = completed + 1\n                         WHERE run_id = ?;', (task_group_id,))
        conn.commit()