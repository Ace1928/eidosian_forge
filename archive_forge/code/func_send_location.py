import json
import os
import random
import time
import copy
import numpy as np
import pickle
from joblib import Parallel, delayed
from parlai.core.worlds import MultiAgentDialogWorld
from parlai.mturk.core.agents import MTURK_DISCONNECT_MESSAGE
from parlai.mturk.core.worlds import MTurkOnboardWorld
def send_location(self, agent):
    """
        Sends the current location to the given agent.
        """
    msg = {'id': 'WORLD_LOCATION', 'message_id': 'WORLD_LOCATION', 'text': {'location': self.location, 'boundaries': [self.min_x, self.min_y, self.max_x, self.max_y], 'neighborhood': self.neighborhood}}
    agent.observe(msg)