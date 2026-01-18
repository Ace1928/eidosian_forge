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
def set_optimizer_type(self, optimizer):
    optimizer_name = optimizer.__class__.__name__.lower()
    if 'adam' in optimizer_name:
        self.megatron_lm_default_args['optimizer'] = 'adam'
        self.megatron_lm_default_args['adam_beta1'] = optimizer.defaults['betas'][0]
        self.megatron_lm_default_args['adam_beta2'] = optimizer.defaults['betas'][1]
        self.megatron_lm_default_args['adam_eps'] = optimizer.defaults['eps']
    elif 'sgd' in optimizer_name:
        self.megatron_lm_default_args['optimizer'] = 'sgd'
        self.megatron_lm_default_args['sgd_momentum'] = optimizer.defaults['momentum']
    else:
        raise ValueError(f'Optimizer {optimizer_name} is not supported by Megatron-LM')
    self.megatron_lm_default_args['lr'] = optimizer.defaults['lr']
    self.megatron_lm_default_args['weight_decay'] = optimizer.defaults['weight_decay']