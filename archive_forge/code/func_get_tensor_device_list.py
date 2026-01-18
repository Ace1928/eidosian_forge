import numpy
from ray.util.collective.types import ReduceOp, torch_available
def get_tensor_device_list(tensors):
    """Returns the gpu devices of the list of input tensors.

    Args:
        tensors: a list of tensors, each locates on a GPU.

    Returns:
        list: the list of GPU devices.

    """
    if not isinstance(tensors, list):
        raise RuntimeError("Expect a list of tensors each locates on a GPU device. Got: '{}'.".format(type(tensors)))
    devices = [get_tensor_device(t) for t in tensors]
    return devices