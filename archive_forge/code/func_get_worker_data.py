import json
import logging
import os
import pickle
import sqlite3
import time
import threading
import parlai.mturk.core.dev.shared_utils as shared_utils
from parlai.mturk.core.dev.agents import AssignState
def get_worker_data(self, worker_id):
    """
        get all worker data for a particular worker_id.
        """
    with self.table_access_condition:
        conn = self._get_connection()
        c = conn.cursor()
        c.execute('SELECT * FROM workers WHERE worker_id = ?;', (worker_id,))
        results = c.fetchone()
        return results