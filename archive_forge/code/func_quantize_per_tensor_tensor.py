import torch
from torch.library import Library, impl
from torch.ao.quantization.utils import determine_qparams, validate_qmin_qmax
from typing import Tuple
@impl(quantized_decomposed_lib, 'quantize_per_tensor.tensor', 'CompositeExplicitAutograd')
def quantize_per_tensor_tensor(input: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, quant_min: int, quant_max: int, dtype: torch.dtype) -> torch.Tensor:
    """ Affine quantization for the Tensor using the same quantization parameters to map
    from floating point to quantized values
    Same as `quantize_per_tensor` but scale and zero_point are Scalar Tensor instead of
    scalar values
    """
    assert zero_point.numel() == 1, f'Expecting zero_point tensor to be one element, but received : {zero_point.numel()}'
    assert scale.numel() == 1, f'Expecting scale tensor to be one element, but received : {scale.numel()}'
    return quantize_per_tensor(input, scale.item(), zero_point.item(), quant_min, quant_max, dtype)