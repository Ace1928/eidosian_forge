import json
import logging
import os
import pickle
import sqlite3
import time
import threading
import parlai.mturk.core.dev.shared_utils as shared_utils
from parlai.mturk.core.dev.agents import AssignState
def get_hit_data(self, hit_id):
    """
        get the hit data for the given hit_id, return None if not.
        """
    with self.table_access_condition:
        conn = self._get_connection()
        c = conn.cursor()
        c.execute('SELECT * FROM hits WHERE hit_id = ?;', (hit_id,))
        results = c.fetchone()
        return results