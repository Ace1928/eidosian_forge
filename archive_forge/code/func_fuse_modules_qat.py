import copy
import torch.nn as nn
from torch.ao.quantization.fuser_method_mappings import get_fuser_method
from torch.ao.quantization.fuser_method_mappings import fuse_conv_bn  # noqa: F401
from torch.ao.quantization.fuser_method_mappings import fuse_conv_bn_relu  # noqa: F401
from torch.nn.utils.parametrize import type_before_parametrizations
from typing import List, Optional
def fuse_modules_qat(model, modules_to_fuse, inplace=False, fuser_func=fuse_known_modules, fuse_custom_config_dict=None):
    """QAT version for `fuse_modules`."""
    return _fuse_modules(model, modules_to_fuse, is_qat=True, inplace=inplace, fuser_func=fuser_func, fuse_custom_config_dict=fuse_custom_config_dict)