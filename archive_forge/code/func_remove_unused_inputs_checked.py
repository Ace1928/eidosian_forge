import torch.fx as fx
import copy
import torch
import math
import sys
from typing import Callable, List
from functools import wraps, partial
from dataclasses import dataclass
from .compile_utils import get_placeholders, get_outputs
from torch.utils._content_store import ContentStoreWriter
from torch.hub import tqdm
from torch.multiprocessing.reductions import StorageWeakRef
import os
def remove_unused_inputs_checked(cur_state: ReproState):
    new_state = remove_unused_inputs_unchecked(cur_state)
    if new_state is not None and graph_fails(new_state.graph, new_state.inps):
        return new_state
    return None