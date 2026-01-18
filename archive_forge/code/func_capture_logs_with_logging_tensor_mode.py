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
@contextlib.contextmanager
def capture_logs_with_logging_tensor_mode(python_tb=False, script_tb=False, cpp_tb=False):
    with LoggingTensorMode(), capture_logs(True, python_tb, script_tb, cpp_tb) as logs:
        yield logs