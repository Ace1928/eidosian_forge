import torch
from collections import OrderedDict
import weakref
import warnings
from typing import Any, Tuple
def setup_input_hook(self, args):

    def fn(grad_fn):
        self._set_user_hook(grad_fn)
    res, input_idx = self._apply_on_tensors(fn, args)
    self.n_inputs = len(args)
    self.input_tensors_index = input_idx
    return res