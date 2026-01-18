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
def sufficient_overlap(self, text, sent_dict):
    text_list = [w[:4] for w in split_tokenize(text.lower()) if w not in STOPWORDS]
    for _, sentence in sent_dict.items():
        sentence_list = [w[:4] for w in split_tokenize(sentence.lower()) if w not in STOPWORDS]
        if len(set(text_list).intersection(set(sentence_list))) >= self.opt.get('word_overlap_threshold', 2):
            return True
    return False