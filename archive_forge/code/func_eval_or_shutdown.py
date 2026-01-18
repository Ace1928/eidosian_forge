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
def eval_or_shutdown(agent):
    if self.model_agent is None and agent == self.other_agent:
        control_msg = self.get_control_msg()
        control_msg['text'] = OTHER_AGENT_FINISHED_MSG
        self.other_agent.observe(validate(control_msg))
        self.eval_agent.mturk_manager.mark_workers_done([self.eval_agent])
        self.other_agent.shutdown()
    else:
        evaluations = [self.evaluate_engagingness, self.evaluate_interestingness, self.evaluate_inquisitiveness, self.evaluate_listening, self.evaluate_repetitiveness, self.evaluate_fluency, self.evaluate_consistency, self.evaluate_humanness, self.evaluate_persona]
        for evaluation in evaluations:
            fin = evaluation()
            if not fin:
                return
        return