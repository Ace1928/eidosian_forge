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
def is_msg_tooshortlong(self, act, ag, th_min=3, th_max=20):
    if act['episode_done']:
        return False
    control_msg = self.get_control_msg()
    msg_len = len(act['text'].split(' '))
    if msg_len < th_min:
        control_msg['text'] = TOO_SHORT_MSG.format(th_min)
        ag.observe(validate(control_msg))
        return True
    if msg_len > th_max:
        control_msg['text'] = TOO_LONG_MSG.format(th_max)
        ag.observe(validate(control_msg))
        return True
    return False