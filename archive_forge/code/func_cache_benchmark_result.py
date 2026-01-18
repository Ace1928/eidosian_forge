import copy
import itertools
import logging
from typing import Callable, Optional
from torch.utils._triton import has_triton
from .utils import red_text, triton_config_to_hashable
from . import config as inductor_config
def cache_benchmark_result(self, config, timing):
    self.cached_benchmark_results[triton_config_to_hashable(config)] = timing