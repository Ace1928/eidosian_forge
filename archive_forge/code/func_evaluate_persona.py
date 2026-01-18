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
def evaluate_persona(self):
    if self.model_agent is not None:
        other_persona = self.model_personas
    else:
        other_persona = self.other_agent.personas
    fake_persona = self.eval_agent.personas_generator.get_persona()
    while fake_persona == other_persona:
        fake_persona = self.eval_agent.personas_generator.get_persona()
    cand_text = []
    for dt in [other_persona, fake_persona]:
        if dt == other_persona:
            is_correct = True
        else:
            is_correct = False
        _text = ''
        for s in dt:
            _text += '<b><span style="color:blue">' + s.strip() + '</span></b><br>'
        cand_text.append((is_correct, _text))
    random.shuffle(cand_text)
    control_msg = self.get_control_msg()
    control_msg['text'] = PERSONA_MSG.format(cand_text[0][1], cand_text[1][1])
    control_msg['button_choices'] = '</ROUND>'.join(PERSONA_CHOICES)
    self.eval_agent.observe(validate(control_msg))
    act = self.eval_agent.act(timeout=self.max_resp_time)
    timeout = self.check_timeout(act)
    if timeout:
        return False
    self.persona_scores.append(cand_text[int(act['text']) - 1][0])
    return True