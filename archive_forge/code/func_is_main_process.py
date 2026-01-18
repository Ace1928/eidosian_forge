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
def is_main_process(local_rank):
    """
    Whether or not the current process is the local process, based on `xm.get_ordinal()` (for TPUs) first, then on
    `local_rank`.
    """
    if is_torch_tpu_available(check_device=True):
        import torch_xla.core.xla_model as xm
        return xm.get_ordinal() == 0
    return local_rank in [-1, 0]