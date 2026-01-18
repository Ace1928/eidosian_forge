import collections
from typing import Any, Callable, Dict, Optional
import torch
import torch.utils._pytree as pytree
def add_use(inp):
    if inp in seen_uses:
        return
    seen_uses.add(inp)
    last_non_output_use[node].append(inp)