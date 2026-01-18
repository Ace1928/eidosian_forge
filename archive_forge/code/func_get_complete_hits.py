import logging
import os
import pickle
import time
from parlai.mturk.core.dev.agents import MTurkAgent, AssignState
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
def get_complete_hits(self):
    """
        Returns the list of all currently completed HITs.
        """
    hit_ids = []
    for hit_id, agent in self.hit_id_to_agent.items():
        if agent.hit_is_complete:
            hit_ids.append(hit_id)
    return hit_ids