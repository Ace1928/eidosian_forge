import numpy
from ray.util.collective.types import ReduceOp, torch_available
def get_nccl_tensor_dtype(tensor):
    """Return the corresponded NCCL dtype given a tensor."""
    if isinstance(tensor, cupy.ndarray):
        return NUMPY_NCCL_DTYPE_MAP[tensor.dtype.type]
    if torch_available():
        if isinstance(tensor, torch.Tensor):
            return TORCH_NCCL_DTYPE_MAP[tensor.dtype]
    raise ValueError('Unsupported tensor type. Got: {}. Supported GPU tensor types are: torch.Tensor, cupy.ndarray.'.format(type(tensor)))