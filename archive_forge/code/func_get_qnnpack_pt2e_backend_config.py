import operator
import torch
from torch.ao.quantization.backend_config import (
from typing import List
def get_qnnpack_pt2e_backend_config():
    return BackendConfig('qnnpack_pytorch_2.0_export').set_backend_pattern_configs(get_linear_configs()).set_backend_pattern_configs(get_binary_op_configs()).set_backend_pattern_configs(get_conv_configs()).set_backend_pattern_configs(get_pooling_configs()).set_backend_pattern_configs(get_relu_configs())