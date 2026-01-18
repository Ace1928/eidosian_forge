import json
import logging
import os
import pickle
import sqlite3
import time
import threading
import parlai.mturk.core.dev.shared_utils as shared_utils
from parlai.mturk.core.dev.agents import AssignState
def get_all_task_assignments_for_worker(self, worker_id, task_group_id=None):
    """
        get all assignments for a particular worker within a particular run by worker_id
        and task_group_id.
        """
    task_group_id = self._force_task_group_id(task_group_id)
    with self.table_access_condition:
        conn = self._get_connection()
        c = conn.cursor()
        c.execute('SELECT assignments.assignment_id, assignments.status,\n                         assignments.approve_time, assignments.worker_id,\n                         assignments.hit_id\n                         FROM assignments\n                         INNER JOIN hits on assignments.hit_id = hits.hit_id\n                         WHERE assignments.worker_id = ? AND hits.run_id = ?;\n                         ', (worker_id, task_group_id))
        results = c.fetchall()
        return results