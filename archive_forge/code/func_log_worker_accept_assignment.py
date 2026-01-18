import json
import logging
import os
import pickle
import sqlite3
import time
import threading
import parlai.mturk.core.dev.shared_utils as shared_utils
from parlai.mturk.core.dev.agents import AssignState
def log_worker_accept_assignment(self, worker_id, assignment_id, hit_id, task_group_id=None):
    """
        Log a worker accept, update assignment state and pairings to match the
        acceptance.
        """
    task_group_id = self._force_task_group_id(task_group_id)
    with self.table_access_condition:
        conn = self._get_connection()
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM workers WHERE worker_id = ?;', (worker_id,))
        has_worker = c.fetchone()[0] > 0
        if not has_worker:
            c.execute('INSERT INTO workers VALUES (?,?,?,?,?,?,?);', (worker_id, 1, 0, 0, 0, 0, 0))
        else:
            c.execute('UPDATE workers SET accepted = accepted + 1\n                             WHERE worker_id = ?;', (worker_id,))
        c.execute('REPLACE INTO assignments VALUES (?,?,?,?,?)', (assignment_id, 'Accepted', None, worker_id, hit_id))
        c.execute('INSERT INTO pairings\n                         VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', (AssignState.STATUS_NONE, None, None, None, None, None, 0, '', False, '', worker_id, assignment_id, task_group_id, None, 0, ''))
        conn.commit()