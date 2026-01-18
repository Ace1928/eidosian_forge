import torch
from torch.ao.quantization.qconfig import QConfig
from torch.ao.quantization.quant_type import QuantType
from torch.jit._recursive import wrap_cpp_module
def fuse_conv_bn_jit(model, inplace=False):
    """ Fuse conv - bn module
    Works for eval model only.

    Args:
        model: TorchScript model from scripting or tracing
    """
    torch._C._log_api_usage_once('quantization_api.quantize_jit.fuse_conv_bn_jit')
    model_c = model._c
    model_c = torch._C._jit_pass_fold_convbn(model_c)
    if inplace:
        model._reconstruct(model_c)
    else:
        model = wrap_cpp_module(model_c)
    return model