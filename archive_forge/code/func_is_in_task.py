import logging
import time
from queue import Queue
import uuid
from parlai.core.agents import Agent
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
def is_in_task(self):
    """
        Simple check for an agent being in a task.
        """
    return self.get_status() == AssignState.STATUS_IN_TASK