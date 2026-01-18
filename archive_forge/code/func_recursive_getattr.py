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
def recursive_getattr(obj, attr: str):
    """
    Recursive `getattr`.

    Args:
        obj:
            A class instance holding the attribute.
        attr (`str`):
            The attribute that is to be retrieved, e.g. 'attribute1.attribute2'.
    """

    def _getattr(obj, attr):
        return getattr(obj, attr)
    return reduce(_getattr, [obj] + attr.split('.'))