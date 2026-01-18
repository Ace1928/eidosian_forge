import operator
from typing import List
import torch
import torch.ao.nn.qat as nnqat
import torch.ao.nn.quantized.reference as nnqr
import torch.nn as nn
import torch.nn.functional as F
from ..fuser_method_mappings import (
from ._common_operator_config_utils import _Conv2dMetadata
from .backend_config import (
from .qnnpack import (
def get_executorch_backend_config() -> BackendConfig:
    """
    Return the `BackendConfig` for backends PyTorch lowers to through the Executorch stack.
    """
    return BackendConfig('executorch').set_backend_pattern_configs(_get_linear_configs()).set_backend_pattern_configs(_get_conv_configs()).set_backend_pattern_configs(_get_binary_ops_configs()).set_backend_pattern_configs(_get_share_qparams_ops_configs()).set_backend_pattern_configs(_get_bn_configs()).set_backend_pattern_configs(_get_cat_configs()).set_backend_pattern_configs(_get_embedding_op_configs())