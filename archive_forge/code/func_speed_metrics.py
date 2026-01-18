import copy
import functools
import gc
import inspect
import os
import random
import re
import threading
import time
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
import numpy as np
from .utils import (
def speed_metrics(split, start_time, num_samples=None, num_steps=None, num_tokens=None):
    """
    Measure and return speed performance metrics.

    This function requires a time snapshot `start_time` before the operation to be measured starts and this function
    should be run immediately after the operation to be measured has completed.

    Args:
    - split: name to prefix metric (like train, eval, test...)
    - start_time: operation start time
    - num_samples: number of samples processed
    - num_steps: number of steps processed
    - num_tokens: number of tokens processed
    """
    runtime = time.time() - start_time
    result = {f'{split}_runtime': round(runtime, 4)}
    if runtime == 0:
        return result
    if num_samples is not None:
        samples_per_second = num_samples / runtime
        result[f'{split}_samples_per_second'] = round(samples_per_second, 3)
    if num_steps is not None:
        steps_per_second = num_steps / runtime
        result[f'{split}_steps_per_second'] = round(steps_per_second, 3)
    if num_tokens is not None:
        tokens_per_second = num_tokens / runtime
        result[f'{split}_tokens_per_second'] = round(tokens_per_second, 3)
    return result