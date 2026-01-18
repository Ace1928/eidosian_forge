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
def assemble_tensormap_to_arg(self, args):
    args_with_tma = list(args)
    if hasattr(self, 'tensormaps_info'):
        args_ptr = tuple([arg.data_ptr() if hasattr(arg, 'data_ptr') else arg for arg in args])
        for i, e in enumerate(self.tensormaps_info):
            args_with_tma.append(CompiledKernel.tensormap_manager[e, args_ptr])
    return args_with_tma