import collections
import logging
import os
import random
import types
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import torch
from packaging.version import Version
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import (
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
from ray.train._internal import session
from ray.train._internal.accelerator import Accelerator
from ray.train._internal.session import get_accelerator, set_accelerator
from ray.util.annotations import Deprecated, PublicAPI
def model_get_state(self):
    if hasattr(self, '_original_get_state'):
        state = self._original_get_state()
        state['__getstate__'] = state['_original_get_state']
        del state['_original_get_state']
    else:
        state = self.__dict__.copy()
        del state['__getstate__']
    state['forward'] = state['_unwrapped_forward']
    del state['_unwrapped_forward']
    return state