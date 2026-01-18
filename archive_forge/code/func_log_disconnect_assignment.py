import json
import logging
import os
import pickle
import sqlite3
import time
import threading
import parlai.mturk.core.dev.shared_utils as shared_utils
from parlai.mturk.core.dev.agents import AssignState
def log_disconnect_assignment(self, worker_id, assignment_id, approve_time, disconnect_type, task_group_id=None):
    """
        Note that an assignment was disconnected from.
        """
    task_group_id = self._force_task_group_id(task_group_id)
    with self.table_access_condition:
        conn = self._get_connection()
        c = conn.cursor()
        c.execute('UPDATE assignments SET status = ?, approve_time = ?\n                         WHERE assignment_id = ?;', ('Disconnected', approve_time, assignment_id))
        c.execute('UPDATE workers SET disconnected = disconnected + 1\n                         WHERE worker_id = ?;', (worker_id,))
        c.execute('UPDATE pairings SET status = ?, task_end = ?\n                         WHERE worker_id = ? AND assignment_id = ?;', (disconnect_type, time.time(), worker_id, assignment_id))
        c.execute('UPDATE runs SET failed = failed + 1 WHERE run_id = ?;', (task_group_id,))
        conn.commit()