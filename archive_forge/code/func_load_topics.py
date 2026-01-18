from parlai.core.agents import create_agent_from_shared
from parlai.core.message import Message
from parlai.core.worlds import validate, MultiAgentDialogWorld
from parlai.mturk.core.agents import TIMEOUT_MESSAGE
import parlai.mturk.core.mturk_utils as mutils
from parlai.mturk.core.worlds import MTurkOnboardWorld
from parlai.tasks.wizard_of_wikipedia.build import build
from parlai.tasks.wizard_of_wikipedia.agents import TOKEN_KNOWLEDGE, TOKEN_END_KNOWLEDGE
from joblib import Parallel, delayed
import json
import numpy as np
import os
import pickle
import random
import time
def load_topics(self):
    if not os.path.isfile(self.topics_path):
        build(self.opt)
    with open(self.topics_path) as f:
        self.data = json.load(f)
    self.seen_topics = self.data['train']
    self.unseen_topics = self.data['valid'] + self.data['test']