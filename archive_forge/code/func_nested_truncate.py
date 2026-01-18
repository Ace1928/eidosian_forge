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
def nested_truncate(tensors, limit):
    """Truncate `tensors` at `limit` (even if it's a nested list/tuple/dict of tensors)."""
    if isinstance(tensors, (list, tuple)):
        return type(tensors)((nested_truncate(t, limit) for t in tensors))
    if isinstance(tensors, Mapping):
        return type(tensors)({k: nested_truncate(t, limit) for k, t in tensors.items()})
    return tensors[:limit]