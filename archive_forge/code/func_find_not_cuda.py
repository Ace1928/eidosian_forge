import torch
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.utils import _pytree as pytree
import operator
def find_not_cuda(t):
    nonlocal found_not_cuda
    if isinstance(t, torch.Tensor) and t.device.type != 'cuda':
        found_not_cuda = True