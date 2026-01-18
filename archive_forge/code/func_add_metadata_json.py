import gzip
import json
import os
import tempfile
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from warnings import warn
import torch
import torch.autograd.profiler as prof
from torch._C import _get_privateuse1_backend_name
from torch._C._profiler import (
from torch.autograd import kineto_available, ProfilerActivity
from torch.profiler._memory_profiler import MemoryProfile, MemoryProfileTimeline
def add_metadata_json(self, key: str, value: str):
    """
        Adds a user defined metadata with a string key and a valid json value
        into the trace file
        """
    torch.autograd._add_metadata_json(key, value)