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
def check_timeout(self, act):
    if act is None:
        self.chat_done = True
        return True
    if act['text'] == '[TIMEOUT]' or act['text'] == '[RETURNED]' or act['text'] == '[DISCONNECT]':
        control_msg = self.get_control_msg()
        control_msg['episode_done'] = True
        control_msg['text'] = self.get_instruction(agent_id=act['id'], tag='timeout')
        for ag in self.agents:
            if ag.id != act['id']:
                ag.observe(validate(control_msg))
        self.chat_done = True
        return True
    else:
        return False