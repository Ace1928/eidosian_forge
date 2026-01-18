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
def set_auto_wrap_policy(self, model):
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
    default_transformer_cls_names_to_wrap = ','.join(model._no_split_modules) if getattr(model, '_no_split_modules', None) is not None else ''
    if self.auto_wrap_policy is None:
        auto_wrap_policy = os.environ.get('FSDP_AUTO_WRAP_POLICY', 'NO_WRAP')
        if auto_wrap_policy == FSDP_AUTO_WRAP_POLICY[0]:
            transformer_cls_names_to_wrap = os.environ.get('FSDP_TRANSFORMER_CLS_TO_WRAP', default_transformer_cls_names_to_wrap).split(',')
            transformer_cls_to_wrap = set()
            for layer_class in transformer_cls_names_to_wrap:
                transformer_cls = FullyShardedDataParallelPlugin.get_module_class_from_name(model, layer_class)
                if transformer_cls is None:
                    raise Exception('Could not find the transformer layer class to wrap in the model.')
                else:
                    transformer_cls_to_wrap.add(transformer_cls)
            self.auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=transformer_cls_to_wrap)
        elif auto_wrap_policy == FSDP_AUTO_WRAP_POLICY[1]:
            min_num_params = int(os.environ.get('FSDP_MIN_NUM_PARAMS', 0))
            if min_num_params > 0:
                self.auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=min_num_params)