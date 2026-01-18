import enum
import os
import socket
import subprocess
import uuid
from platform import uname
from typing import List, Tuple, Union
from packaging.version import parse, Version
import psutil
import torch
import asyncio
from functools import partial
from typing import (
from collections import OrderedDict
from typing import Any, Hashable, Optional
from vllm.logger import init_logger
def get_nvcc_cuda_version() -> Optional[Version]:
    cuda_home = os.environ.get('CUDA_HOME')
    if not cuda_home:
        cuda_home = '/usr/local/cuda'
        if os.path.isfile(cuda_home + '/bin/nvcc'):
            logger.info(f'CUDA_HOME is not found in the environment. Using {cuda_home} as CUDA_HOME.')
        else:
            logger.warning(f'Not found nvcc in {cuda_home}. Skip cuda version check!')
            return None
    nvcc_output = subprocess.check_output([cuda_home + '/bin/nvcc', '-V'], universal_newlines=True)
    output = nvcc_output.split()
    release_idx = output.index('release') + 1
    nvcc_cuda_version = parse(output[release_idx].split(',')[0])
    return nvcc_cuda_version