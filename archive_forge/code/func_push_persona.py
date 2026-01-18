from parlai.mturk.core.worlds import MTurkOnboardWorld
from parlai.mturk.core.agents import TIMEOUT_MESSAGE
from parlai.core.worlds import validate, MultiAgentDialogWorld
from joblib import Parallel, delayed
from extract_and_save_personas import main as main_extract
import numpy as np
import time
import os
import pickle
import random
def push_persona(self, idx):
    self.idx_stack.append(idx)