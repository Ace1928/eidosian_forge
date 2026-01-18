import dataclasses
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type
import torch
from torch._export import ExportedProgram
from torch.utils._pytree import (
def default_to_dumpable_context(context: Context) -> DumpableContext:
    return (serialized_type, context[1], context[2])