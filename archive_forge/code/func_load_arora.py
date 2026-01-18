from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.core.build_data import modelzoo_path
import torchtext.vocab as vocab
from parlai.utils.misc import TimeLogger
from collections import Counter, deque
import numpy as np
import os
import pickle
import torch
def load_arora(opt):
    """
    Load the data in the arora.pkl file in data/controllable_dialogue.
    """
    arora_fp = os.path.join(opt['datapath'], CONTROLLABLE_DIR, 'arora.pkl')
    print('Loading Arora embedding info from %s...' % arora_fp)
    with open(arora_fp, 'rb') as f:
        data = pickle.load(f)
    print('Done loading arora info.')
    return data