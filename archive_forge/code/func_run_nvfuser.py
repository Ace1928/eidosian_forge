from contextlib import contextmanager
from typing import Any, List, Tuple, cast
import random
import torch
import time
from torch.utils.benchmark import Timer
def run_nvfuser(ir, inputs) -> float:
    with torch.jit.fuser('fuser2'):
        return run_test(ir, inputs)