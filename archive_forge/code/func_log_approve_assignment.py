import json
import logging
import os
import pickle
import sqlite3
import time
import threading
import parlai.mturk.core.dev.shared_utils as shared_utils
from parlai.mturk.core.dev.agents import AssignState
def log_approve_assignment(self, assignment_id):
    """
        Update assignment state to reflect approval, update worker state to increment
        number of accepted assignments.
        """
    with self.table_access_condition:
        conn = self._get_connection()
        c = conn.cursor()
        c.execute('SELECT * FROM assignments WHERE assignment_id = ?;', (assignment_id,))
        assignment = c.fetchone()
        if assignment is None:
            return
        status = assignment['status']
        worker_id = assignment['worker_id']
        if status == 'Approved':
            return
        c.execute('UPDATE assignments SET status = ?\n                         WHERE assignment_id = ?;', ('Approved', assignment_id))
        if status == 'Rejected':
            c.execute('UPDATE workers SET approved = approved + 1,\n                             rejected = rejected - 1\n                             WHERE worker_id = ?;', (worker_id,))
        else:
            c.execute('UPDATE workers SET approved = approved + 1\n                             WHERE worker_id = ?;', (worker_id,))
        conn.commit()