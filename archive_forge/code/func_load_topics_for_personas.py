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
def load_topics_for_personas(self):
    self.persona_to_topics = {}
    with open(self.topic_for_personas_path) as f:
        text = f.read()
        personas = text.split('\n\n')
        for persona in personas:
            persona = persona.split('\n')
            prev_p = persona[0]
            for i in range(1, len(persona)):
                p_i = persona[i]
                if 'https' in p_i:
                    topic = unquote(p_i[p_i.rfind('/') + 1:]).replace('_', ' ')
                    if prev_p in self.persona_to_topics:
                        self.persona_to_topics[prev_p].append(topic)
                    else:
                        self.persona_to_topics[prev_p] = [topic]
                else:
                    prev_p = p_i