import ctypes as ct
from functools import reduce  # Required in Python 3
import itertools
import operator
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
from torch import Tensor
from bitsandbytes.utils import pack_dict_to_tensor, unpack_tensor_to_dict
from .cextension import COMPILED_WITH_CUDA, lib
def dequantize_4bit(A: Tensor, quant_state: Optional[QuantState]=None, absmax: Optional[torch.Tensor]=None, out: Optional[torch.Tensor]=None, blocksize: int=64, quant_type='fp4') -> Tensor:
    """
    Dequantizes FP4 blockwise quantized values.

    Dequantizes the tensor A with maximum absolute values absmax in blocks of size blocksize.

    Parameters
    ----------
    A : torch.Tensor
        The input tensor (packed 4-bit values).
    quant_state : QuantState
        object with quantisation stats, incl. absmax values, original tensor shape and original dtype.
    absmax : torch.Tensor
        The absmax values.
    out : torch.Tensor
        Dequantized output tensor.
    blocksize : int
        The blocksize used in quantization.
    quant_type : str
        The 4-bit quantization data type {fp4, nf4}


    Returns
    -------
    torch.Tensor:
        Dequantized tensor.
    """
    if blocksize not in [2048, 4096, 1024, 512, 256, 128, 64]:
        raise ValueError(f'The blockwise of {blocksize} is not supported. Supported values: [2048, 4096, 1024, 512, 256, 128, 64]')
    if quant_type not in ['fp4', 'nf4']:
        raise NotImplementedError(f'4-bit quantization data type {quant_type} is not implemented.')
    if quant_state is None:
        assert absmax is not None and out is not None
        quant_state = QuantState(absmax=absmax, shape=out.shape, dtype=out.dtype, blocksize=blocksize, quant_type=quant_type)
    else:
        absmax = quant_state.absmax
    if quant_state.nested:
        absmax = dequantize_blockwise(quant_state.absmax, quant_state.state2)
        absmax += quant_state.offset
        if absmax.dtype != torch.float32:
            absmax = absmax.float()
    if out is None:
        out = torch.empty(quant_state.shape, dtype=quant_state.dtype, device=A.device)
    n = out.numel()
    device = pre_call(A.device)
    is_on_gpu([A, absmax, out])
    if out.dtype == torch.float32:
        if quant_state.quant_type == 'fp4':
            lib.cdequantize_blockwise_fp32_fp4(get_ptr(None), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int(quant_state.blocksize), ct.c_int(n))
        else:
            lib.cdequantize_blockwise_fp32_nf4(get_ptr(None), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int(quant_state.blocksize), ct.c_int(n))
    elif out.dtype == torch.float16:
        if quant_state.quant_type == 'fp4':
            lib.cdequantize_blockwise_fp16_fp4(get_ptr(None), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int(quant_state.blocksize), ct.c_int(n))
        else:
            lib.cdequantize_blockwise_fp16_nf4(get_ptr(None), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int(quant_state.blocksize), ct.c_int(n))
    elif out.dtype == torch.bfloat16:
        if quant_state.quant_type == 'fp4':
            lib.cdequantize_blockwise_bf16_fp4(get_ptr(None), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int(quant_state.blocksize), ct.c_int(n))
        else:
            lib.cdequantize_blockwise_bf16_nf4(get_ptr(None), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int(quant_state.blocksize), ct.c_int(n))
    else:
        raise ValueError(f'Blockwise quantization only supports 16/32-bit floats, but got {A.dtype}')
    post_call(A.device)
    is_transposed = True if A.shape[0] == 1 else False
    if is_transposed:
        return out.t()
    else:
        return out