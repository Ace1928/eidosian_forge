import torch
import torch.fx
import warnings
import functools
import builtins
from typing import Any, Callable, Dict, Optional, Union
def functional_relu_override(x, inplace=False):
    assert not inplace, 'dont support inplace functional.relu for metatensor analysis'
    return x