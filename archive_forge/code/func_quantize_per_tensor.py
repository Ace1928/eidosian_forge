import torch
from torch.library import Library, impl
from torch.ao.quantization.utils import determine_qparams, validate_qmin_qmax
from typing import Tuple
@impl(quantized_decomposed_lib, 'quantize_per_tensor', 'CompositeExplicitAutograd')
def quantize_per_tensor(input: torch.Tensor, scale: float, zero_point: int, quant_min: int, quant_max: int, dtype: torch.dtype) -> torch.Tensor:
    """ Affine quantization for the Tensor using the same quantization parameters to map
    from floating point to quantized values

    Args:
       input (torch.Tensor): original float32 or bfloat16 Tensor
       scale (float): quantization parameter for affine quantization
       zero_point (int): quantization parameter for affine quantization
       quant_min (int): minimum quantized value for output Tensor
       quant_max (int): maximum quantized value for output Tensor
       dtype (torch.dtype): requested dtype (e.g. torch.uint8) for output Tensor

    Returns:
       Tensor with requested dtype (e.g. torch.uint8), note the quantization parameters
       are not stored in the Tensor, we are storing them in function arguments instead
    """
    if input.dtype == torch.bfloat16:
        input = input.to(torch.float32)
    assert input.dtype == torch.float32, f'Expecting input to have dtype torch.float32, but got dtype: {input.dtype}'
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    inv_scale = 1.0 / scale
    return torch.clamp(torch.round(input * inv_scale) + zero_point, quant_min, quant_max).to(dtype)