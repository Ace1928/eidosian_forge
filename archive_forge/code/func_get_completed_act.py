import logging
import time
from queue import Queue
import uuid
from parlai.core.agents import Agent
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
def get_completed_act(self):
    """
        Returns completed act upon arrival, errors on disconnect.
        """
    while self.completed_message is None:
        self.assert_connected()
        time.sleep(shared_utils.THREAD_SHORT_SLEEP)
    return self.completed_message