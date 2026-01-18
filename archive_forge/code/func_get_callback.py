import collections
from contextlib import contextmanager
from typing import List, Tuple
import torch
import torch.fx.traceback as fx_traceback
def get_callback(saved_stack_):

    def callback():
        global callback_set
        fx_traceback.set_stack_trace(saved_stack_)
        callback_set = False
    return callback