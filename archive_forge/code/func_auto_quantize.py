import functools
import torch
import torch.distributed as dist
from enum import Enum
def auto_quantize(func, qtype, quant_loss=None):
    """
    Quantize the input tensors, choose the precision types, and pass other necessary arguments and then dequantizes the output.

    Currently it only supports:
        . FP16 and BFP16 quantization method supported for gloo and nccl backends
        . all_gather, all_to_all collective ops
    Note: BFP16 only supports 2D tensors.
    Args:
        func (Callable): A function representing collective operations.
        qtype (QuantType): Quantization method
        quant_loss (float, optional): This can be used to improve accuracy in the dequantization.
    Returns:
        (Callable): the same collective as func but enables automatic quantization/dequantization.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        group = kwargs.get('group', None)
        async_op = kwargs.get('async_op', False)
        if async_op is True:
            raise RuntimeError('The async_op=True mode is not supported yet.')
        if func == dist.all_gather:
            tensors = args[0]
            input_tensors = _quantize_tensor(args[1], qtype)
            out_tensors = _quantize_tensor_list(tensors, qtype)
            dist.all_gather(out_tensors, input_tensors, group=group, async_op=async_op)
            for i, t in enumerate(_dequantize_tensor_list(out_tensors, qtype, quant_loss=quant_loss)):
                tensors[i] = t
        elif func == dist.all_to_all:
            tensors = args[0]
            input_tensors = _quantize_tensor_list(args[1], qtype)
            out_tensors = _quantize_tensor_list(tensors, qtype)
            dist.all_to_all(out_tensors, input_tensors, group=group, async_op=async_op)
            for i, t in enumerate(_dequantize_tensor_list(out_tensors, qtype, quant_loss=quant_loss)):
                tensors[i] = t
        elif func == dist.all_to_all_single:
            tensors = args[0]
            out_splits = kwargs.get('out_splits', None)
            in_splits = kwargs.get('in_splits', None)
            input_tensors = _quantize_tensor(args[1], qtype)
            out_tensors = _quantize_tensor(tensors, qtype)
            dist.all_to_all_single(out_tensors, input_tensors, out_splits, in_splits, group=group)
            for i, t in enumerate(_dequantize_tensor(out_tensors, qtype, quant_loss=quant_loss)):
                tensors[i] = t
        else:
            raise RuntimeError(f'The collective op {func} is not supported yet')
    return wrapper