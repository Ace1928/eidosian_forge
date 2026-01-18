import torch
from .backend_config import (
from ._common_operator_config_utils import (
def get_tensorrt_backend_config_dict():
    """
    Return the `BackendConfig` for the TensorRT backend in dictionary form.
    """
    return get_tensorrt_backend_config().to_dict()