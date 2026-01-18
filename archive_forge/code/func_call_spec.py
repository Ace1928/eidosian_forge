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
@property
@compatibility(is_backward_compatible=False)
def call_spec(self):
    from torch._export.exported_program import CallSpec
    if len(self.module_call_graph) == 0:
        return CallSpec(in_spec=None, out_spec=None)
    assert self.module_call_graph[0].fqn == ''
    return CallSpec(in_spec=self.module_call_graph[0].signature.in_spec, out_spec=self.module_call_graph[0].signature.out_spec)