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
def extract_personas(self):
    personas = []
    with open(self.text_file, 'r') as f:
        lines = f.readlines()
    new_persona = []
    for line in lines:
        if 'persona: ' in line:
            new_persona.append(line.split('persona: ')[1].replace('\n', ''))
        elif new_persona:
            personas.append(new_persona)
            new_persona = []
    return personas