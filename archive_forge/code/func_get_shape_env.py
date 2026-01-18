import copy
import dataclasses
import functools
from typing import (
import torch
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree
from torch.fx._compatibility import compatibility
from torch.fx.experimental.proxy_tensor import maybe_disable_fake_tensor_mode
from torch.fx.passes.infra.pass_base import PassResult
from torch.fx.passes.infra.pass_manager import PassManager
from .graph_signature import (  # noqa: F401
def get_shape_env(gm):
    vals = [node.meta['val'] for node in gm.graph.nodes if node.meta.get('val', None) is not None]
    from torch._guards import detect_fake_mode
    fake_mode = detect_fake_mode(vals)
    if fake_mode is not None:
        return fake_mode.shape_env
    for v in vals:
        if isinstance(v, torch.SymInt):
            return v.node.shape_env