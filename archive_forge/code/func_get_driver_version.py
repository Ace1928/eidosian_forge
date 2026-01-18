import logging
import math
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from shutil import which
from typing import List, Optional
import torch
from packaging.version import parse
def get_driver_version():
    """
    Returns the driver version

    In the case of multiple GPUs, will return the first.
    """
    output = subprocess.check_output([_nvidia_smi(), '--query-gpu=driver_version', '--format=csv,noheader'], universal_newlines=True)
    output = output.strip()
    return output.split(os.linesep)[0]