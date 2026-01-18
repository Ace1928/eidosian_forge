from collections import defaultdict
from typing import Any, Dict, List, Optional
from warnings import warn
import torch
import torch.cuda
from torch._C import _get_privateuse1_backend_name
from torch._C._profiler import _ExperimentalConfig
from torch.autograd import (
from torch.autograd.profiler_util import (
from torch.futures import Future
def load_nvprof(path):
    """Open an nvprof trace file and parses autograd annotations.

    Args:
        path (str): path to nvprof trace
    """
    return EventList(parse_nvprof_trace(path))