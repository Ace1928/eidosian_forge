import copy
import itertools
import logging
from typing import Callable, Optional
from torch.utils._triton import has_triton
from .utils import red_text, triton_config_to_hashable
from . import config as inductor_config
def get_warpsmax(self):
    return 1024 // 32