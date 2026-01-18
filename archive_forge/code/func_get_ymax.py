import copy
import itertools
import logging
from typing import Callable, Optional
from torch.utils._triton import has_triton
from .utils import red_text, triton_config_to_hashable
from . import config as inductor_config
def get_ymax(self):
    ymax = inductor_config.triton.max_block['Y']
    if self.size_hints and len(self.size_hints) > 1:
        ymax = min(ymax, self.size_hints[1])
    return ymax