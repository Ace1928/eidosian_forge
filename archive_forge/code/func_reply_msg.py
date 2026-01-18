import os, copy, types, gc, sys
import numpy as np
from prompt_toolkit import prompt
import torch
from rwkv.model import RWKV
from rwkv.utils import PIPELINE
def reply_msg(msg):
    print(f'{bot}{interface} {msg}\n')