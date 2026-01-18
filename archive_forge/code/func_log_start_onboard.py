import json
import logging
import os
import pickle
import sqlite3
import time
import threading
import parlai.mturk.core.dev.shared_utils as shared_utils
from parlai.mturk.core.dev.agents import AssignState
def log_start_onboard(self, worker_id, assignment_id, conversation_id):
    """
        Update a pairing state to reflect onboarding status.
        """
    with self.table_access_condition:
        conn = self._get_connection()
        c = conn.cursor()
        c.execute('UPDATE pairings SET status = ?, onboarding_start = ?,\n                         onboarding_id = ?\n                         WHERE worker_id = ? AND assignment_id = ?;', (AssignState.STATUS_ONBOARDING, time.time(), conversation_id, worker_id, assignment_id))
        conn.commit()