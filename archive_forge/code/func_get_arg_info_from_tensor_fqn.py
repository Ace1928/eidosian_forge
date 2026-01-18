from typing import Any, Dict, Optional, Type
from torch.nn.utils.parametrize import type_before_parametrizations, is_parametrized
from itertools import chain
from torch import nn
def get_arg_info_from_tensor_fqn(model: nn.Module, tensor_fqn: str) -> Dict[str, Any]:
    """
    Uses tensor_fqn to obtain a dict containing module_fqn, module and tensor_name
    """
    tensor_name = tensor_fqn.split('.')[-1]
    module_fqn = tensor_fqn[:-len(tensor_name) - ('.' in tensor_fqn)]
    module = fqn_to_module(model, module_fqn)
    return {'module_fqn': module_fqn, 'module': module, 'tensor_name': tensor_name, 'tensor_fqn': tensor_fqn}