import json
import logging
import os
import pickle
import sqlite3
import time
import threading
import parlai.mturk.core.dev.shared_utils as shared_utils
from parlai.mturk.core.dev.agents import AssignState
def log_finish_onboard(self, worker_id, assignment_id):
    """
        Update a pairing state to reflect waiting status.
        """
    with self.table_access_condition:
        conn = self._get_connection()
        c = conn.cursor()
        c.execute('UPDATE pairings SET status = ?, onboarding_end = ?\n                         WHERE worker_id = ? AND assignment_id = ?;', (AssignState.STATUS_WAITING, time.time(), worker_id, assignment_id))
        conn.commit()