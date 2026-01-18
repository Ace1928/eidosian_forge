import torch
from torch.ao.quantization.qconfig import QConfig
from torch.ao.quantization.quant_type import QuantType
from torch.jit._recursive import wrap_cpp_module
def prepare_dynamic_jit(model, qconfig_dict, inplace=False):
    torch._C._log_api_usage_once('quantization_api.quantize_jit.prepare_dynamic_jit')
    return _prepare_jit(model, qconfig_dict, inplace, quant_type=QuantType.DYNAMIC)