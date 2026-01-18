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
def nested_numpify(tensors):
    """Numpify `tensors` (even if it's a nested list/tuple/dict of tensors)."""
    if isinstance(tensors, (list, tuple)):
        return type(tensors)((nested_numpify(t) for t in tensors))
    if isinstance(tensors, Mapping):
        return type(tensors)({k: nested_numpify(t) for k, t in tensors.items()})
    t = tensors.cpu()
    if t.dtype == torch.bfloat16:
        t = t.to(torch.float32)
    return t.numpy()