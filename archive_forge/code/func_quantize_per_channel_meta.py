import torch
from torch.library import Library, impl
from torch.ao.quantization.utils import determine_qparams, validate_qmin_qmax
from typing import Tuple
@impl(quantized_decomposed_lib, 'quantize_per_channel', 'Meta')
def quantize_per_channel_meta(input: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor, axis: int, quant_min: int, quant_max: int, dtype: torch.dtype) -> torch.Tensor:
    assert input.dtype == torch.float32, f'Expecting input to have dtype torch.float32, but got dtype: {input.dtype}'
    assert axis < input.dim(), f'Expecting axis to be < {input.dim()}'
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    return torch.empty_like(input, dtype=dtype)