import collections
import os
import platform
import re
import socket
from contextlib import contextmanager
from functools import partial, reduce
from types import MethodType
from typing import OrderedDict
import torch
from packaging.version import Version
from safetensors.torch import save_file as safe_save_file
from ..commands.config.default import write_basic_config  # noqa: F401
from ..logging import get_logger
from ..state import PartialState
from .constants import FSDP_PYTORCH_VERSION
from .dataclasses import DistributedType
from .imports import is_deepspeed_available, is_torch_distributed_available, is_torch_xla_available
from .modeling import id_tensor_storage
from .transformer_engine import convert_model
from .versions import is_torch_version
def check_os_kernel():
    """Warns if the kernel version is below the recommended minimum on Linux."""
    info = platform.uname()
    system = info.system
    if system != 'Linux':
        return
    _, version, *_ = re.split('(\\d+\\.\\d+\\.\\d+)', info.release)
    min_version = '5.5.0'
    if Version(version) < Version(min_version):
        msg = f'Detected kernel version {version}, which is below the recommended minimum of {min_version}; this can cause the process to hang. It is recommended to upgrade the kernel to the minimum version or higher.'
        logger.warning(msg, main_process_only=True)