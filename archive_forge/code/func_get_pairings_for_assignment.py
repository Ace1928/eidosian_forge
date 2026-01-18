import json
import logging
import os
import pickle
import sqlite3
import time
import threading
import parlai.mturk.core.dev.shared_utils as shared_utils
from parlai.mturk.core.dev.agents import AssignState
def get_pairings_for_assignment(self, assignment_id):
    """
        get all pairings attached to a particular assignment_id.
        """
    with self.table_access_condition:
        conn = self._get_connection()
        c = conn.cursor()
        c.execute('SELECT * FROM pairings WHERE assignment_id = ?;', (assignment_id,))
        results = c.fetchall()
        return results