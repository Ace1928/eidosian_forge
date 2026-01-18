import os
import platform
import subprocess
import sys
from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple, Union
from tqdm.auto import tqdm
from cmdstanpy import _DOT_CMDSTAN
from .. import progress as progbar
from .logging import get_logger
def determine_linux_arch() -> str:
    machine = platform.machine()
    arch = ''
    if machine == 'aarch64':
        arch = 'arm64'
    elif machine == 'armv7l':
        readelf = subprocess.run(['readelf', '-A', '/proc/self/exe'], check=True, stdout=subprocess.PIPE, text=True)
        if 'Tag_ABI_VFP_args' in readelf.stdout:
            arch = 'armel'
        else:
            arch = 'armhf'
    elif machine == 'mips64':
        arch = 'mips64el'
    elif machine == 'ppc64el' or machine == 'ppc64le':
        arch = 'ppc64el'
    elif machine == 's390x':
        arch = 's390x'
    return arch