from typing import Optional
import torch.fx
from torch.fx import Node
from torch.fx._compatibility import compatibility
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
from torch.fx.experimental.proxy_tensor import py_sym_types, snapshot_fake
from torch.fx.node import map_aggregate
def propagate_dont_convert_inputs(self, *args):
    with self._mode:
        return super().run(*args)