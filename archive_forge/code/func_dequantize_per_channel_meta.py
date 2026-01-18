import torch
from torch.library import Library, impl
from torch.ao.quantization.utils import determine_qparams, validate_qmin_qmax
from typing import Tuple
@impl(quantized_decomposed_lib, 'dequantize_per_channel', 'Meta')
def dequantize_per_channel_meta(input: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor, axis: int, quant_min: int, quant_max: int, dtype: torch.dtype) -> torch.Tensor:
    assert input.dtype == dtype, f'Expecting input to have dtype {dtype}, but got dtype: {input.dtype}'
    assert axis < input.dim(), f'Expecting axis to be < {input.dim()}'
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    return torch.empty_like(input, dtype=torch.float32)