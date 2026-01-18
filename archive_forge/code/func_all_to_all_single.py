import torch
import torch.distributed as dist
from torch.autograd import Function
from torch.distributed import group, ReduceOp
def all_to_all_single(output, input, output_split_sizes=None, input_split_sizes=None, group=group.WORLD):
    """
    Each process splits input tensor and then scatters the split list to all processes in a group.

    Then concatenate the received tensors from all the processes in the group and return single output tensor.

    Arguments:
        output (Tensor): Gathered concatenated output tensor.
        input (Tensor): Input tensor to scatter.
        output_split_sizes: (list[Int], optional): Output split sizes for dim 0
            if specified None or empty, dim 0 of ``output`` tensor must divide
            equally by ``world_size``.
        input_split_sizes: (list[Int], optional): Input split sizes for dim 0
            if specified None or empty, dim 0 of ``input`` tensor must divide
            equally by ``world_size``.

    Returns:
        Tensor: Output of the collective.

    """
    return _AlltoAllSingle.apply(group, output, output_split_sizes, input_split_sizes, input)