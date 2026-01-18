import json
import logging
import os
import pickle
import sqlite3
import time
import threading
import parlai.mturk.core.dev.shared_utils as shared_utils
from parlai.mturk.core.dev.agents import AssignState
def log_pay_extra_bonus(self, worker_id, assignment_id, amount, reason):
    """
        Update a pairing state to add a bonus to be paid, appends reason.

        To be used for extra bonuses awarded at the discretion of the requester
        """
    with self.table_access_condition:
        conn = self._get_connection()
        c = conn.cursor()
        reason = '${} for {}\n'.format(amount, reason)
        cent_amount = int(amount * 100)
        c.execute('UPDATE pairings\n                        SET extra_bonus_amount = extra_bonus_amount + ?,\n                        extra_bonus_text = extra_bonus_text || ?\n                         WHERE worker_id = ? AND assignment_id = ?;', (cent_amount, reason, worker_id, assignment_id))
        conn.commit()