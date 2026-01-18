import torch
import torch.fx
import warnings
import functools
import builtins
from typing import Any, Callable, Dict, Optional, Union
def embedding_override(self, input):
    return torch.empty(*input.shape, self.weight.shape[-1], device='meta')