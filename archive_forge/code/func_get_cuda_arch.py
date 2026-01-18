import functools
import logging
from typing import Optional
import torch
from ... import config
def get_cuda_arch() -> Optional[str]:
    try:
        cuda_arch = config.cuda.arch
        if cuda_arch is None:
            major, minor = torch.cuda.get_device_capability(0)
            cuda_arch = major * 10 + minor
        return str(cuda_arch)
    except Exception as e:
        log.error('Error getting cuda arch: %s', e)
        return None