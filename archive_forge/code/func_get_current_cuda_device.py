import logging
from typing import Optional
import torch
def get_current_cuda_device():
    current_device = str(torch.cuda.get_device_properties(torch.cuda.current_device()))
    for device_str in SUPPORTED_CUDA_DEVICES:
        if current_device.find(device_str) > 0:
            return device_str
    logger.warning('Unsupported device, Triton code generation may fail')
    return 'P100'