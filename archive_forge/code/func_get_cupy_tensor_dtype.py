import numpy
from ray.util.collective.types import ReduceOp, torch_available
def get_cupy_tensor_dtype(tensor):
    """Return the corresponded Cupy dtype given a tensor."""
    if isinstance(tensor, cupy.ndarray):
        return tensor.dtype.type
    if torch_available():
        if isinstance(tensor, torch.Tensor):
            return TORCH_NUMPY_DTYPE_MAP[tensor.dtype]
    raise ValueError('Unsupported tensor type. Got: {}. Supported GPU tensor types are: torch.Tensor, cupy.ndarray.'.format(type(tensor)))