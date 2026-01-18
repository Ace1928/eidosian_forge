import logging
import os
import pickle
import time
from parlai.mturk.core.dev.agents import MTurkAgent, AssignState
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
def has_assignment(self, assign_id):
    """
        Returns true if this worker has an assignment for the given id.
        """
    return assign_id in self.agents