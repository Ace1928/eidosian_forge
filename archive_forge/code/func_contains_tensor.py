import os
import warnings
from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd
import torch
import ray
from ray.air.util.data_batch_conversion import _unwrap_ndarray_object_type_if_needed
def contains_tensor(obj):
    if isinstance(obj, torch.Tensor):
        return True
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if contains_tensor(k):
                return True
            if contains_tensor(v):
                return True
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            if contains_tensor(v):
                return True
    return False