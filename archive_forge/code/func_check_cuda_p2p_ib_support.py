import os
import platform
import subprocess
import sys
from shutil import which
from typing import List
import torch
def check_cuda_p2p_ib_support():
    """
    Checks if the devices being used have issues with P2P and IB communications, namely any consumer GPU hardware after
    the 3090.

    Noteably uses `nvidia-smi` instead of torch to not initialize CUDA.
    """
    try:
        device_names, device_count = get_gpu_info()
        unsupported_devices = {'RTX 40'}
        if device_count > 1:
            if any((unsupported_device in device_name for device_name in device_names for unsupported_device in unsupported_devices)):
                return False
    except Exception:
        pass
    return True