import torch
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.utils import _pytree as pytree
import operator
def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
    if node.op not in CALLABLE_NODE_OPS:
        return False
    if node.target in [torch.ops.aten.embedding_dense_backward.default]:
        return False
    if node.target in [operator.getitem]:
        return True
    found_not_cuda = False

    def meta_fk(meta):
        return meta['val'] if 'val' in meta else meta['fake_result']

    def find_not_cuda(t):
        nonlocal found_not_cuda
        if isinstance(t, torch.Tensor) and t.device.type != 'cuda':
            found_not_cuda = True
    for n in node.all_input_nodes:
        pytree.tree_map_(find_not_cuda, meta_fk(n.meta))
    pytree.tree_map_(find_not_cuda, meta_fk(node.meta))
    return not found_not_cuda