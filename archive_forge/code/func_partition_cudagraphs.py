import torch
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.utils import _pytree as pytree
import operator
def partition_cudagraphs(gm, inputs):
    """
    Partition an FX graph into sub-GraphModules that can be validly run under
    CUDA graphs.  For a subgraph to be runnable under CUDA, all of the operations
    must involve CUDA tensors only/
    """
    FakeTensorProp(gm).propagate(*inputs)
    supported_ops = CudaGraphsSupport()
    partitioner = CapabilityBasedPartitioner(gm, supported_ops, allows_single_node_partition=True)
    partitions = partitioner.propose_partitions()
    fused_graph = partitioner.fuse_partitions(partitions)
    return fused_graph