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
def proxy_call_hook(self, hook, *args):
    return self.fx_tracer.create_proxy('call_function', call_hook, (hook, *[self.to_proxy(x) for x in args]), {})