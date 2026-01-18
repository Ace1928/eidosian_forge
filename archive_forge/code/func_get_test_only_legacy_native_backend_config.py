import torch
from ._common_operator_config_utils import (
from .backend_config import BackendConfig, DTypeConfig
def get_test_only_legacy_native_backend_config() -> BackendConfig:
    """
    Return the `BackendConfig` for PyTorch Native backend (fbgemm/qnnpack) with various additional fp16 ops.
    """
    conv_dtype_configs = [weighted_op_quint8_dtype_config]
    linear_dtype_configs = [weighted_op_quint8_dtype_config, default_dynamic_int8_dtype_config, default_dynamic_float16_dtype_config, default_op_fp16_dtype_config]
    binary_op_dtype_configs = [default_op_quint8_dtype_config, default_op_fp16_dtype_config]
    default_op_dtype_configs = [default_op_quint8_dtype_config]
    fixed_qparams_op_dtype_configs = [default_op_quint8_dtype_config, default_op_fp16_dtype_config]
    share_qparams_op_dtype_configs = [default_op_quint8_dtype_config, default_op_fp16_dtype_config]
    tensor_info_op_dtype_configs = [default_op_quint8_dtype_config]
    rnn_op_dtype_configs = [default_dynamic_int8_dtype_config, default_dynamic_float16_dtype_config]
    embedding_op_dtype_configs = [weight_only_quint8_dtype_config, weight_only_quint4x2_dtype_config]
    layer_norm_op_dtype_configs = [input_output_only_quint8_dtype_config]
    return BackendConfig('_native_and_fp16').set_backend_pattern_configs(_get_conv_configs(conv_dtype_configs)).set_backend_pattern_configs(_get_linear_configs(linear_dtype_configs)).set_backend_pattern_configs(_get_binary_op_configs(binary_op_dtype_configs)).set_backend_pattern_config(_get_cat_config(default_op_dtype_configs)).set_backend_pattern_configs(_get_default_op_configs(default_op_dtype_configs)).set_backend_pattern_configs(_get_fixed_qparams_op_configs(fixed_qparams_op_dtype_configs)).set_backend_pattern_configs(_get_share_qparams_op_configs(share_qparams_op_dtype_configs)).set_backend_pattern_configs(_get_tensor_info_op_configs(tensor_info_op_dtype_configs)).set_backend_pattern_configs(_get_bn_configs(default_op_dtype_configs)).set_backend_pattern_configs(_get_ln_configs(layer_norm_op_dtype_configs)).set_backend_pattern_configs(_get_rnn_op_configs(rnn_op_dtype_configs)).set_backend_pattern_configs(_get_embedding_op_configs(embedding_op_dtype_configs))