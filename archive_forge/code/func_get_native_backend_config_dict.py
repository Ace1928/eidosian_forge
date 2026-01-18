import torch
from ._common_operator_config_utils import (
from .backend_config import BackendConfig, DTypeConfig
def get_native_backend_config_dict():
    """
    Return the `BackendConfig` for PyTorch Native backend (fbgemm/qnnpack) in dictionary form.
    """
    return get_native_backend_config().to_dict()