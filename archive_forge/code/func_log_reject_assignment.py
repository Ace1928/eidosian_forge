import json
import logging
import os
import pickle
import sqlite3
import time
import threading
import parlai.mturk.core.dev.shared_utils as shared_utils
from parlai.mturk.core.dev.agents import AssignState
def log_reject_assignment(self, assignment_id):
    """
        Update assignment state to reflect rejection, update worker state to increment
        number of rejected assignments.
        """
    with self.table_access_condition:
        conn = self._get_connection()
        c = conn.cursor()
        c.execute('UPDATE assignments SET status = ?\n                         WHERE assignment_id = ? AND status != ?;', ('Rejected', assignment_id, 'Rejected'))
        if c.rowcount > 0:
            c.execute('SELECT * FROM assignments WHERE assignment_id = ?;', (assignment_id,))
            assignment = c.fetchone()
            if assignment is None:
                return
            worker_id = assignment['worker_id']
            c.execute('UPDATE workers SET rejected = rejected + 1\n                             WHERE worker_id = ?;', (worker_id,))
        conn.commit()