import argparse
import copy
import enum
import functools
import os
import typing
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, get_args
import torch
from .constants import FSDP_AUTO_WRAP_POLICY, FSDP_BACKWARD_PREFETCH, FSDP_SHARDING_STRATEGY, FSDP_STATE_DICT_TYPE
from .environment import str_to_bool
from .imports import is_cuda_available, is_npu_available, is_xpu_available
from .versions import compare_versions
def deepspeed_config_process(self, prefix='', mismatches=None, config=None, must_match=True, **kwargs):
    """Process the DeepSpeed config with the values from the kwargs."""
    mismatches = [] if mismatches is None else mismatches
    if config is None:
        config = self.deepspeed_config
    for key, value in config.items():
        if isinstance(value, dict):
            self.deepspeed_config_process(prefix=prefix + key + '.', mismatches=mismatches, config=value, must_match=must_match, **kwargs)
        else:
            self.fill_match(prefix + key, mismatches, must_match=must_match, **kwargs)
    if len(mismatches) > 0 and prefix == '':
        mismatches_msg = '\n'.join(mismatches)
        raise ValueError(f"Please correct the following DeepSpeed config values that mismatch kwargs  values:\n{mismatches_msg}\nThe easiest method is to set these DeepSpeed config values to 'auto'.")