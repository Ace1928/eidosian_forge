import gc
from typing import Any
import torch
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
def is_cudnn_snafu(exception: BaseException) -> bool:
    return isinstance(exception, RuntimeError) and len(exception.args) == 1 and ('cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.' in exception.args[0])