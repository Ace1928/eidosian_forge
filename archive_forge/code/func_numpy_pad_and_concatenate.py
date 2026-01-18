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
def numpy_pad_and_concatenate(array1, array2, padding_index=-100):
    """Concatenates `array1` and `array2` on first axis, applying padding on the second if necessary."""
    array1 = atleast_1d(array1)
    array2 = atleast_1d(array2)
    if len(array1.shape) == 1 or array1.shape[1] == array2.shape[1]:
        return np.concatenate((array1, array2), axis=0)
    new_shape = (array1.shape[0] + array2.shape[0], max(array1.shape[1], array2.shape[1])) + array1.shape[2:]
    result = np.full_like(array1, padding_index, shape=new_shape)
    result[:array1.shape[0], :array1.shape[1]] = array1
    result[array1.shape[0]:, :array2.shape[1]] = array2
    return result