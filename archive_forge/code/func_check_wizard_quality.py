from parlai.mturk.core.worlds import MTurkOnboardWorld
from parlai.mturk.core.agents import TIMEOUT_MESSAGE
from parlai.core.worlds import validate, MultiAgentDialogWorld
from parlai.utils.safety import OffensiveStringMatcher
from joblib import Parallel, delayed
from task_config import task_config as config
from extract_and_save_personas import main as main_extract
from constants import (
import numpy as np
import time
import os
import pickle
import random
import copy
from urllib.parse import unquote
def check_wizard_quality(self):
    """
        Determines whether to soft-block this turker or not Only called if the
        conversation finishes Returns True if the Wizard is good.
        """
    num_good_sents = len(list(filter(lambda info: 'good_message' in info and info['good_message'], self.dialog)))
    wizard_worker = [w for w in self.agents if w.id == WIZARD][0].worker_id
    data_path = self.opt['current_working_dir']
    bad_wizards = os.path.join(data_path, 'bad_wizards.txt')
    good_wizards = os.path.join(data_path, 'good_wizards.txt')
    if num_good_sents < self.opt['num_good_sentence_threshold']:
        if not self.opt['is_sandbox']:
            with open(bad_wizards, 'a') as f:
                f.write(wizard_worker + '\n')
        return (False, wizard_worker)
    else:
        if not self.opt['is_sandbox']:
            with open(good_wizards, 'a') as f:
                f.write(wizard_worker + '\n')
        return (True, wizard_worker)