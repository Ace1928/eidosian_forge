import torch
from torch.ao.quantization.qconfig import QConfig
from torch.ao.quantization.quant_type import QuantType
from torch.jit._recursive import wrap_cpp_module
def script_qconfig(qconfig):
    """Instantiate the activation and weight observer modules and script
    them, these observer module instances will be deepcopied during
    prepare_jit step.
    """
    return QConfig(activation=torch.jit.script(qconfig.activation())._c, weight=torch.jit.script(qconfig.weight())._c)