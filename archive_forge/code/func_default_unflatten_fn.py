import dataclasses
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type
import torch
from torch._export import ExportedProgram
from torch.utils._pytree import (
def default_unflatten_fn(values: Iterable[Any], context: Context) -> Any:
    typ, flat_names, none_names = context
    return typ(**dict(zip(flat_names, values)), **{k: None for k in none_names})