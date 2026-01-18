import os, copy, types, gc, sys
import numpy as np
from prompt_toolkit import prompt
import torch
from rwkv.model import RWKV
from rwkv.utils import PIPELINE
def save_all_stat(srv, name, last_out):
    n = f'{name}_{srv}'
    all_state[n] = {}
    all_state[n]['out'] = last_out
    all_state[n]['rnn'] = copy.deepcopy(model_state)
    all_state[n]['token'] = copy.deepcopy(model_tokens)