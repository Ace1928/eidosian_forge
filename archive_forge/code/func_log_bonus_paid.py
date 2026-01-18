import json
import logging
import os
import pickle
import sqlite3
import time
import threading
import parlai.mturk.core.dev.shared_utils as shared_utils
from parlai.mturk.core.dev.agents import AssignState
def log_bonus_paid(self, worker_id, assignment_id):
    """
        Update to show that the intended bonus amount awarded for work in the task has
        been paid.
        """
    with self.table_access_condition:
        conn = self._get_connection()
        c = conn.cursor()
        c.execute('UPDATE pairings SET bonus_paid = ?\n                         WHERE worker_id = ? AND assignment_id = ?;', (True, worker_id, assignment_id))
        conn.commit()