import functools
from torch._dynamo.device_interface import get_interface_for_device
def is_cuda_compatible_with_triton():
    device_interface = get_interface_for_device('cuda')
    return device_interface.is_available() and device_interface.Worker.get_device_properties().major >= 7