import copy
import importlib.metadata
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from packaging import version
from ..utils import is_auto_awq_available, is_torch_available, logging
def post_init(self):
    """
        Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.
        """
    if not isinstance(self.in_group_size, int):
        raise ValueError('in_group_size must be a float')
    if not isinstance(self.out_group_size, int):
        raise ValueError('out_group_size must be a float')
    if not isinstance(self.num_codebooks, int):
        raise ValueError('num_codebooks must be a float')
    if not isinstance(self.nbits_per_codebook, int):
        raise ValueError('nbits_per_codebook must be a float')
    if self.linear_weights_not_to_quantize is not None and (not isinstance(self.linear_weights_not_to_quantize, list)):
        raise ValueError('linear_weights_not_to_quantize must be a list of strings')
    if self.linear_weights_not_to_quantize is None:
        self.linear_weights_not_to_quantize = []