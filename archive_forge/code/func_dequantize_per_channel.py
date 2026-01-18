import torch
from torch.library import Library, impl
from torch.ao.quantization.utils import determine_qparams, validate_qmin_qmax
from typing import Tuple
@impl(quantized_decomposed_lib, 'dequantize_per_channel', 'CompositeExplicitAutograd')
def dequantize_per_channel(input: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor, axis: int, quant_min: int, quant_max: int, dtype: torch.dtype) -> torch.Tensor:
    """ Affine per channel dequantization for the Tensor using the same quantization
    parameters for each channel/axis to map from quantized values to floating point values

    Args:
       input (torch.Tensor): Tensor with dtype matching `dtype` argument,
       e.g. (`torch.uint8`), it is a per channel quantized Tensor if combined with
       quantization parameter in the argument of this function (scales/zero_points/axis)

       scales (torch.Tensor): a list of scale quantization parameter for
       affine quantization, one per channel

       zero_points (torch.Tensor): a list of zero_point quantization parameter for
       affine quantization, one per channel

       quant_min (int): minimum quantized value for output Tensor (not used in computation,
       reserved for pattern matching)

       quant_max (int): maximum quantized value for output Tensor (not used in computation,
       reserved for pattern matching)

       dtype (torch.dtype): requested dtype for output Tensor (not used in computation,
       reserved for pattern matching)

    Returns:
       dequantized float32 Tensor
    """
    assert input.dtype == dtype, f'Expecting input to have dtype {dtype}, but got dtype: {input.dtype}'
    assert axis < input.dim(), f'Expecting axis to be < {input.dim()}'
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    input, permute_axis_list = _permute_to_axis_zero(input, axis)
    res = torch.zeros_like(input, dtype=torch.float32)
    for i in range(input.size(0)):
        res[i] = (input[i].to(torch.float32) - zero_points[i]) * scales[i]
    out = res.permute(tuple(permute_axis_list))
    return out