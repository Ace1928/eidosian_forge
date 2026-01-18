from __future__ import annotations
import functools
import hashlib
import json
import os
import re
from collections import namedtuple
from pathlib import Path
from typing import Any
from dataclasses import dataclass
from .._C.libtriton.triton import (ClusterInfo, TMAInfos, add_external_libs, compile_ptx_to_cubin, get_env_vars,
from ..common.backend import get_backend, get_cuda_version_key, path_to_ptxas
from ..common.build import is_hip
from ..runtime.autotuner import OutOfResources
from ..runtime.cache import get_cache_manager, get_dump_manager, get_override_manager
from ..runtime.driver import driver
from ..runtime.jit import (JITFunction, get_cuda_stream, get_current_device, get_device_capability)
from ..tools.disasm import get_sass
from .code_generator import ast_to_ttir
from .make_launcher import make_stub
from .utils import (InfoFromBackendForTensorMap, TensorMapManager, get_ids_of_tensormaps, parse_tma_info)
def llir_to_ptx(mod: Any, target: CudaTargetDescriptor, ptx_version: int=None) -> str:
    """
    Translate TritonGPU module to PTX code.
    :param mod: a TritonGPU dialect module
    :return: PTX code
    """
    if ptx_version is None:
        _, cuda_version = path_to_ptxas()
        ptx_version = ptx_get_version(cuda_version)
    return translate_llvmir_to_ptx(mod, target.capability, ptx_version, target.enable_fp_fusion)