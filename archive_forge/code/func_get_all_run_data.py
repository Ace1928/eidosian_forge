import json
import logging
import os
import pickle
import sqlite3
import time
import threading
import parlai.mturk.core.dev.shared_utils as shared_utils
from parlai.mturk.core.dev.agents import AssignState
def get_all_run_data(self, start=0, count=1000):
    """
        get all the run data for all task_group_ids.
        """
    with self.table_access_condition:
        conn = self._get_connection()
        c = conn.cursor()
        c.execute('SELECT * FROM runs LIMIT ?,?;', (start, start + count))
        results = c.fetchall()
        return results