from typing import Any, Dict, List, Mapping
import numpy as np
import torch
from ...utils import is_cython_available, requires_backends
def convert_to_single_emb(x, offset: int=512):
    feature_num = x.shape[1] if len(x.shape) > 1 else 1
    feature_offset = 1 + np.arange(0, feature_num * offset, offset, dtype=np.int64)
    x = x + feature_offset
    return x