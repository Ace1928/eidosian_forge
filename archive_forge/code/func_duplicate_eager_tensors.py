import copy
import dataclasses
import itertools
import os
from typing import Any, Callable, Dict, List
import torch
import torch._lazy as lazy
import torch._lazy.metrics as metrics
from torch import fx
from torch._lazy import computation, debug as lazy_debug
from torch._lazy.tensor_factory_functions import tensor_factory_functions
def duplicate_eager_tensors(self, eager_tensor_list):
    duplicated_list = [None] * self.total_count
    assert len(eager_tensor_list) == len(self.index)
    for uniq_idx, eager_tensor in enumerate(eager_tensor_list):
        for dup_idx in self.index[uniq_idx]:
            duplicated_list[dup_idx] = eager_tensor
    return duplicated_list