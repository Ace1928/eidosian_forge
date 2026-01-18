import json
import logging
import os
import pickle
import sqlite3
import time
import threading
import parlai.mturk.core.dev.shared_utils as shared_utils
from parlai.mturk.core.dev.agents import AssignState
def log_submit_assignment(self, worker_id, assignment_id):
    """
        To be called whenever a worker hits the "submit hit" button.
        """
    with self.table_access_condition:
        conn = self._get_connection()
        c = conn.cursor()
        c.execute('UPDATE assignments SET status = ?\n                         WHERE assignment_id = ?;', ('Reviewable', assignment_id))
        conn.commit()