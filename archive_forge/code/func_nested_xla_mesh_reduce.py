import copy
import datetime
import io
import json
import math
import os
import sys
import warnings
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from logging import StreamHandler
from typing import Any, Dict, Iterator, List, Optional, Union
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import Dataset, IterableDataset, RandomSampler, Sampler
from torch.utils.data.distributed import DistributedSampler
from .integrations.deepspeed import is_deepspeed_zero3_enabled
from .tokenization_utils_base import BatchEncoding
from .utils import is_sagemaker_mp_enabled, is_torch_tpu_available, is_training_run_on_sagemaker, logging
def nested_xla_mesh_reduce(tensors, name):
    if is_torch_tpu_available():
        import torch_xla.core.xla_model as xm
        if isinstance(tensors, (list, tuple)):
            return type(tensors)((nested_xla_mesh_reduce(t, f'{name}_{i}') for i, t in enumerate(tensors)))
        if isinstance(tensors, Mapping):
            return type(tensors)({k: nested_xla_mesh_reduce(t, f'{name}_{i}') for i, (k, t) in enumerate(tensors.items())})
        tensors = atleast_1d(tensors)
        return xm.mesh_reduce(name, tensors, torch.cat)
    else:
        raise ImportError('Torch xla must be installed to use `nested_xla_mesh_reduce`')