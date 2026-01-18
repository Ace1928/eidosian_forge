import torch
from torch.utils._pytree import tree_map
from typing import Iterator, List, Optional
import logging
import contextlib
import itertools
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.weak import WeakTensorKeyDictionary
import functools
from torch._C._profiler import gather_traceback, symbolize_tracebacks
def log_input(name: str, var: object):
    logging.getLogger('LoggingTensor').info('input', (name,), {}, var)