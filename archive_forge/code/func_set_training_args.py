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
def set_training_args(self, micro_batch_size, dp_degree):
    self.data_parallel_size = dp_degree
    self.micro_batch_size = micro_batch_size
    self.global_batch_size = dp_degree * micro_batch_size * self.num_micro_batches
    self.megatron_lm_default_args['data_parallel_size'] = self.data_parallel_size
    self.megatron_lm_default_args['micro_batch_size'] = self.micro_batch_size
    self.megatron_lm_default_args['global_batch_size'] = self.global_batch_size