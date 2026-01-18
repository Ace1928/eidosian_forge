from parlai import __file__ as parlai_filepath
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.mturk.tasks.acute_eval.run import AcuteEvaluator, add_args as acute_add_args
from parlai.scripts.self_chat import self_chat, setup_args as self_chat_setup_args
from parlai.utils.conversations import Conversations, Conversation
from parlai.utils.strings import normalize_reply
from parlai.utils.testing import capture_output
from parlai.mturk.tasks.acute_eval.analysis import (
from parlai.mturk.tasks.acute_eval.dump_task_to_acute_format import (
from parlai.mturk.tasks.acute_eval.configs import CONFIG
from typing import Dict, Any, List, Tuple, Set
from itertools import combinations
import datetime
import time
import json
import os
import random
import torch
import hashlib
def run_acute_eval(self):
    """
        Run ACUTE Eval.
        """
    self._load_pairings_file()
    self.acute_args = acute_add_args()
    self.acute_args.update(ACUTE_DEFAULT_ARGS)
    total_convos = self.opt['matchups_per_pair'] * len(self.combos)
    self.acute_args.update({'is_sandbox': not self.opt['live_acute'], 'pairings_filepath': self.pairings_filepath, 's1_choice': self.question_config['s1_choice'], 's2_choice': self.question_config['s2_choice'], 'question': self.question_config['question'], 'num_matchup_pairs': total_convos, 'num_conversations': int(total_convos / (SUBTASKS_PER_HIT - 1))})
    self.acute_evaluator = AcuteEvaluator(self.acute_args)
    if self.opt['live_acute']:
        self._print_progress('Running ACUTE-EVAL in LIVE Mode')
    else:
        self._print_progress('Running ACUTE-EVAL in SANDBOX Mode')
    self.run_id = self.acute_evaluator.run()