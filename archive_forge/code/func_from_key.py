import inspect
import re
import warnings
from typing import Any, Dict
import torch
from torch.testing import make_tensor
def from_key(key, parameters):
    return dict(zip(sorted(parameters), key))