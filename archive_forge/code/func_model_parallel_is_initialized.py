import contextlib
import torch
from vllm.model_executor.parallel_utils import cupy_utils
def model_parallel_is_initialized():
    """Check if tensor and pipeline parallel groups are initialized."""
    return _TENSOR_MODEL_PARALLEL_GROUP is not None and _PIPELINE_MODEL_PARALLEL_GROUP is not None