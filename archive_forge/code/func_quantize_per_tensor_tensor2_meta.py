import torch
from torch.library import Library, impl
from torch.ao.quantization.utils import determine_qparams, validate_qmin_qmax
from typing import Tuple
@impl(quantized_decomposed_lib, 'quantize_per_tensor.tensor2', 'Meta')
def quantize_per_tensor_tensor2_meta(input, scale, zero_point, quant_min, quant_max, dtype):
    return quantize_per_tensor_tensor_meta(input, scale, zero_point, quant_min, quant_max, dtype)