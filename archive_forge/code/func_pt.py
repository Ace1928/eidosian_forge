import functools
import random
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union
import numpy as np
from transformers.utils import is_tf_available, is_torch_available
from .normalized_config import (
@classmethod
def pt(cls, dtype):
    mapping = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16, 'int64': torch.int64, 'int32': torch.int32, 'int8': torch.int8, 'bool': torch.bool}
    return mapping[dtype]