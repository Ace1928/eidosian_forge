import warnings
from typing import List, Optional, Tuple
import torch
from torch import _VF, Tensor  # noqa: F401
from torch.nn.utils.rnn import PackedSequence
def quantize_linear_modules(module, dtype=torch.int8):
    warnings.warn('quantize_linear_modules function has been deprecated. Please use torch.ao.quantization.quantize_dynamic API instead.')
    reassign = {}
    for name, mod in module.named_modules():
        if mod is module:
            continue
        new_mod = quantize_linear_modules(mod, dtype)
        if new_mod is not mod:
            reassign[name] = new_mod
    for name, mod in reassign.items():
        setattr(module, name, mod)
    if isinstance(module, torch.nn.Linear):
        if dtype == torch.int8:
            return QuantizedLinear(module)
        elif dtype == torch.float16:
            return QuantizedLinearFP16(module)
        else:
            raise RuntimeError(f'Unsupported dtype: {dtype}')
    return module