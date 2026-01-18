import logging
import os
import pickle
import time
from parlai.mturk.core.dev.agents import MTurkAgent, AssignState
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
def disconnected_assignments(self):
    """
        Returns the number of assignments this worker has completed.
        """
    disconnect_count = 0
    for agent in self.agents.values():
        if agent.get_status() in [AssignState.STATUS_DISCONNECT, AssignState.STATUS_RETURNED]:
            disconnect_count += 1
    return disconnect_count