import numpy
from ray.util.collective.types import ReduceOp, torch_available
def get_tensor_device(tensor):
    """Return the GPU index of a tensor."""
    if isinstance(tensor, cupy.ndarray):
        try:
            device = tensor.device.id
        except AttributeError as exec:
            raise RuntimeError('The tensor is not on a valid GPU.') from exec
    elif torch_available() and isinstance(tensor, torch.Tensor):
        device = tensor.device.index
        if not isinstance(device, int):
            raise RuntimeError('The tensor is not on a valid GPU.')
    else:
        raise ValueError('Unsupported tensor type. Got: {}.'.format(type(tensor)))
    return device