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
def metrics_format(self, metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Reformat Trainer metrics values to a human-readable format

    Args:
        metrics (`Dict[str, float]`):
            The metrics returned from train/evaluate/predict

    Returns:
        metrics (`Dict[str, float]`): The reformatted metrics
    """
    metrics_copy = metrics.copy()
    for k, v in metrics_copy.items():
        if '_mem_' in k:
            metrics_copy[k] = f'{v >> 20}MB'
        elif '_runtime' in k:
            metrics_copy[k] = _secs2timedelta(v)
        elif k == 'total_flos':
            metrics_copy[k] = f'{int(v) >> 30}GF'
        elif isinstance(metrics_copy[k], float):
            metrics_copy[k] = round(v, 4)
    return metrics_copy