import json
import logging
import os
import pickle
import sqlite3
import time
import threading
import parlai.mturk.core.dev.shared_utils as shared_utils
from parlai.mturk.core.dev.agents import AssignState
def get_assignments_for_run(self, task_group_id):
    """
        get all assignments for a particular run by task_group_id.
        """
    with self.table_access_condition:
        conn = self._get_connection()
        c = conn.cursor()
        c.execute('SELECT assignments.* FROM assignments\n                         WHERE assignments.hit_id IN (\n                           SELECT hits.hit_id FROM hits\n                           WHERE hits.run_id = ?\n                         );', (task_group_id,))
        results = c.fetchall()
        return results