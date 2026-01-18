import os
import platform
import subprocess
import sys
from shutil import which
from typing import List
import torch
def get_gpu_info():
    """
    Gets GPU count and names using `nvidia-smi` instead of torch to not initialize CUDA.

    Largely based on the `gputil` library.
    """
    if platform.system() == 'Windows':
        command = which('nvidia-smi')
        if command is None:
            command = '%s\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe' % os.environ['systemdrive']
    else:
        command = 'nvidia-smi'
    output = subprocess.check_output([command, '--query-gpu=count,name', '--format=csv,noheader'], universal_newlines=True)
    output = output.strip()
    gpus = output.split(os.linesep)
    gpu_count = len(gpus)
    gpu_names = [gpu.split(',')[1].strip() for gpu in gpus]
    return (gpu_names, gpu_count)