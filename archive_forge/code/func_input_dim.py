from typing import List, Optional
import torch
from vllm.utils import in_wsl
@property
def input_dim(self) -> int:
    raise NotImplementedError()