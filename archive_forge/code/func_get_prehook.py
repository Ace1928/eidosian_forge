import collections
from contextlib import contextmanager
from typing import List, Tuple
import torch
import torch.fx.traceback as fx_traceback
def get_prehook(stack_, seq_nr):

    def prehook(grad_output):
        global callback_set
        if not callback_set:
            torch.autograd.variable.Variable._execution_engine.queue_callback(get_callback(fx_traceback.format_stack()))
            callback_set = True
        fx_traceback.set_stack_trace(stack_)
        fx_traceback.set_grad_fn_seq_nr(seq_nr)
    return prehook