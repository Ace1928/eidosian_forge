from parlai.core.agents import create_agent_from_shared
from parlai.mturk.core.legacy_2018.agents import TIMEOUT_MESSAGE
from parlai.core.worlds import validate, MultiAgentDialogWorld
from parlai.mturk.core.legacy_2018.worlds import MTurkOnboardWorld
from parlai.core.message import Message
from parlai.utils.strings import normalize_reply
from joblib import Parallel, delayed
import numpy as np
import os
import json
import random
import time
import torch
import copy
def save_data(self):
    convo_finished = True
    bad_workers = []
    if self.dialog == [] or self.persona_scores == []:
        convo_finished = False
    self.convo_finished = convo_finished
    data_path = self.opt['save_data_path']
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if convo_finished:
        filename = os.path.join(data_path, '{}_{}_{}.json'.format(time.strftime('%Y%m%d-%H%M%S'), np.random.randint(0, 1000), self.task_type))
    else:
        filename = os.path.join(data_path, '{}_{}_{}_incomplete.json'.format(time.strftime('%Y%m%d-%H%M%S'), np.random.randint(0, 1000), self.task_type))
    json.dump({'dialog': self.dialog, 'dialog_list': self.dialog_list, 'other_first': self.other_first, 'bot_went_first': self.other_first, 'start_time': self.start_time, 'timestamp': time.time(), 'total_time': time.time() - self.start_time, 'workers': [ag.worker_id for ag in self.agents], 'hit_id': [ag.hit_id for ag in self.agents], 'assignment_id': [ag.assignment_id for ag in self.agents], 'human_personas': [ag.personas for ag in self.agents], 'model_personas': self.model_personas, 'bad_workers': bad_workers, 'n_turn': self.n_turn, 'engagingness': self.engagingness_scores, 'interestingness': self.interestingness_scores, 'listening': self.listening_scores, 'consistency': self.consistency_scores, 'inquisitiveness': self.inquisitiveness_scores, 'repetitiveness': self.repetitiveness_scores, 'humanness': self.humanness_scores, 'fluency': self.fluency_scores, 'persona': self.persona_scores, 'opt': self.opt, 'model_config': self.model_config}, open(filename, 'w'))
    print(self.world_tag, ': Data successfully saved at {}.'.format(filename))