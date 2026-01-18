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
def evaluate_location(num_evals, location, target):
    if num_evals == 3:
        return (num_evals, False, True)
    num_evals += 1
    return (num_evals, location[0] == target[0] and location[1] == target[1], False)