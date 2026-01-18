from parlai.core.params import ParlaiParser
from parlai.core.dict import DictionaryAgent
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.utils.misc import TimeLogger
from parlai.core.metrics import normalize_answer
from parlai.core.logs import TensorboardLogger
from controllable_seq2seq.controls import (
from controllable_seq2seq.util import ConvAI2History
from collections import Counter
import copy
import random
import json
import time
import os
def get_word_stats(text, agent_dict, bins=(0, 100, 1000, 100000)):
    """
    Function which takes text sequence and dict, returns word freq and length
    statistics.

    :param sequence: text sequence
    :param agent_dict: can be external dict or dict from the model
    :param bins: list with range boundaries
    :return: freqs dictionary, num words, avg word length, avg char length
    """
    pred_list = agent_dict.tokenize(text)
    pred_freq = [agent_dict.freq[word] for word in pred_list]
    freqs = {i: 0 for i in bins}
    for f in pred_freq:
        for b in bins:
            if f <= b:
                freqs[b] += 1
                break
    wlength = len(pred_list)
    clength = len(text)
    return (freqs, len(pred_freq), wlength, clength)