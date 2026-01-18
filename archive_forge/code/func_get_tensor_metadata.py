import copyreg
import functools
import sys
import traceback
import warnings
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, DefaultDict, List, Optional
import torch
def get_tensor_metadata(tensor):
    assert isinstance(tensor, torch.Tensor)
    return torch._C._get_tensor_metadata(tensor)