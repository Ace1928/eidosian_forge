import json
import logging
import os
import pickle
import sqlite3
import time
import threading
import parlai.mturk.core.dev.shared_utils as shared_utils
from parlai.mturk.core.dev.agents import AssignState
def log_start_task(self, worker_id, assignment_id, conversation_id):
    """
        Update a pairing state to reflect in_task status.
        """
    with self.table_access_condition:
        conn = self._get_connection()
        c = conn.cursor()
        c.execute('UPDATE pairings SET status = ?, task_start = ?,\n                         conversation_id = ? WHERE worker_id = ?\n                         AND assignment_id = ?;', (AssignState.STATUS_IN_TASK, time.time(), conversation_id, worker_id, assignment_id))
        conn.commit()