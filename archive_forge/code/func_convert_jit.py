import torch
from torch.ao.quantization.qconfig import QConfig
from torch.ao.quantization.quant_type import QuantType
from torch.jit._recursive import wrap_cpp_module
def convert_jit(model, inplace=False, debug=False, preserved_attrs=None):
    torch._C._log_api_usage_once('quantization_api.quantize_jit.convert_jit')
    return _convert_jit(model, inplace, debug, quant_type=QuantType.STATIC, preserved_attrs=preserved_attrs)