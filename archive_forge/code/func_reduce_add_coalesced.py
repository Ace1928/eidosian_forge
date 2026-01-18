import warnings
import torch
from torch.cuda import nccl
from torch._utils import _take_tensors, _flatten_dense_tensors, \
from typing import List
def reduce_add_coalesced(inputs, destination=None, buffer_size=10485760):
    """Sum tensors from multiple GPUs.

    Small tensors are first coalesced into a buffer to reduce the number
    of synchronizations.

    Args:
        inputs (Iterable[Iterable[Tensor]]): iterable of iterables that
            contain tensors from a single device.
        destination (int, optional): a device on which the output will be
            placed (default: current device).
        buffer_size (int): maximum size of the buffer used for coalescing

    Returns:
        A tuple of tensors containing an elementwise sum of each group of
        inputs, placed on the ``destination`` device.
    """
    dense_tensors: List[List] = [[] for _ in inputs]
    output = []
    ref_order = []
    for tensor_at_gpus in zip(*inputs):
        if all((t.is_sparse for t in tensor_at_gpus)):
            result = reduce_add(tensor_at_gpus, destination)
            output.append(result)
            ref_order.append(tensor_at_gpus[0])
        else:
            for coll, t in zip(dense_tensors, tensor_at_gpus):
                coll.append(t.to_dense() if t.is_sparse else t)
            ref_order.append(dense_tensors[0][-1])
    itrs = [_take_tensors(tensors, buffer_size) for tensors in dense_tensors]
    for chunks in zip(*itrs):
        flat_tensors = [_flatten_dense_tensors(chunk) for chunk in chunks]
        flat_result = reduce_add(flat_tensors, destination)
        for t in _unflatten_dense_tensors(flat_result, chunks[0]):
            output.append(t.data)
    return tuple(_reorder_tensors_as(output, ref_order))