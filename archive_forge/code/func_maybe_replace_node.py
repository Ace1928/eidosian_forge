import logging
import torch
from torch._export.pass_base import _ExportPassBase
from torch.ao.quantization.pt2e.utils import (
from torch.fx.node import map_arg
from torch.fx.passes.infra.pass_base import PassResult
def maybe_replace_node(n: torch.fx.Node) -> torch.fx.Node:
    if n == dq_node:
        return new_node
    else:
        return n