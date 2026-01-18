import logging
import traceback
from dataclasses import dataclass, field
from typing import Any, List, Optional
import torch
from torch import fx
from torch._dynamo.output_graph import GraphCompileReason
from torch._dynamo.utils import deepcopy_to_fake_tensor, detect_fake_mode
from torch.fx.node import Node
def args_str(args):
    if torch.is_tensor(args):
        return f'T[{args.shape}]'
    elif isinstance(args, tuple):
        return f'tuple({', '.join([args_str(x) for x in args])})'
    elif isinstance(args, list):
        return f'list({', '.join([args_str(x) for x in args])})'
    else:
        return str(args)