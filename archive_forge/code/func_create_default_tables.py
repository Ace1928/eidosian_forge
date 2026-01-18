import json
import logging
import os
import pickle
import sqlite3
import time
import threading
import parlai.mturk.core.dev.shared_utils as shared_utils
from parlai.mturk.core.dev.agents import AssignState
def create_default_tables(self):
    """
        Prepares the default tables in the database if they don't exist.
        """
    with self.table_access_condition:
        conn = self._get_connection()
        c = conn.cursor()
        c.execute(CREATE_RUN_DATA_SQL_TABLE)
        c.execute(CREATE_WORKER_DATA_SQL_TABLE)
        c.execute(CREATE_HIT_DATA_SQL_TABLE)
        c.execute(CREATE_ASSIGN_DATA_SQL_TABLE)
        c.execute(CREATE_PAIRING_DATA_SQL_TABLE)
        updates = ['ALTER TABLE pairings\n                   ADD COLUMN onboarding_id TEXT default null', 'ALTER TABLE runs\n                   ADD COLUMN taskname TEXT default null', 'ALTER TABLE runs\n                   ADD COLUMN launch_time int default 0', 'ALTER TABLE pairings\n                   ADD COLUMN extra_bonus_amount INT default 0', "ALTER TABLE pairings\n                   ADD COLUMN extra_bonus_text TEXT default '';"]
        for update in updates:
            try:
                c.execute(update)
            except sqlite3.Error:
                pass
        conn.commit()