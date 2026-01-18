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
def pop_persona(self):
    if len(self.idx_stack) == 0:
        self.add_idx_stack()
    idx = self.idx_stack.pop()
    data = np.load(os.path.join(self.personas_path, self.personas_name_list[int(idx)]), allow_pickle=True)
    return (idx, data)