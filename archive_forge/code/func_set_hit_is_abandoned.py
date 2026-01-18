import logging
import time
from queue import Queue
import uuid
from parlai.core.agents import Agent
import parlai.mturk.core.data_model as data_model
import parlai.mturk.core.shared_utils as shared_utils
def set_hit_is_abandoned(self):
    """
        Update local state to abandoned and mark the HIT as expired.
        """
    if not self.hit_is_abandoned:
        self.hit_is_abandoned = True
        self.mturk_manager.force_expire_hit(self.worker_id, self.assignment_id)