import contextlib
import functools
from typing import List, Optional
import torch
from torch._dynamo.external_utils import call_hook
from torch._dynamo.source import GetItemSource, LocalSource
from torch._dynamo.utils import counters, lazy_format_graph_code
from torch._logging import getArtifactLogger
from torch._prims_common import clone_preserve_strides
from torch._subclasses import FakeTensorMode
from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import (
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
from torch.fx.proxy import Proxy
def tensor_pre_hook(self, inputs, hook_id, i: int):
    assert self.hooks_proxy is not None
    hook = self.hooks_proxy[hook_id]
    proxy = self.proxy_call_hook(hook, inputs[i])
    with disable_proxy_modes_tracing():
        inputs[i] = maybe_clone(inputs[i])
        self.bind_tensors_to_proxies([inputs[i]], [proxy])
    return inputs