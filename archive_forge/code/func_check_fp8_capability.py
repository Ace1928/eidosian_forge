import os
import platform
import subprocess
import sys
from shutil import which
from typing import List
import torch
def check_fp8_capability():
    """
    Checks if all the current GPUs available support FP8.

    Notably must initialize `torch.cuda` to check.
    """
    cuda_device_capacity = torch.cuda.get_device_capability()
    return cuda_device_capacity >= (8, 9)