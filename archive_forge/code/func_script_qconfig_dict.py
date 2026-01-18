import torch
from torch.ao.quantization.qconfig import QConfig
from torch.ao.quantization.quant_type import QuantType
from torch.jit._recursive import wrap_cpp_module
def script_qconfig_dict(qconfig_dict):
    """Helper function used by `prepare_jit`.
    Apply `script_qconfig` for all entries in `qconfig_dict` that is
    not None.
    """
    return {k: script_qconfig(v) if v else None for k, v in qconfig_dict.items()}