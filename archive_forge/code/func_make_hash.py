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
def make_hash(fn, target, env_vars, device_backend, **kwargs):
    if device_backend is None:
        version_key = get_cuda_version_key()
    else:
        version_key = device_backend.get_version_key()
    if isinstance(fn, JITFunction):
        configs = kwargs['configs']
        signature = kwargs['signature']
        constants = kwargs.get('constants', dict())
        num_warps = kwargs.get('num_warps', 4)
        num_ctas = kwargs.get('num_ctas', 1)
        num_stages = kwargs.get('num_stages', 3)
        enable_warp_specialization = kwargs.get('enable_warp_specialization', False)
        enable_persistent = kwargs.get('enable_persistent', False)
        debug = kwargs.get('debug', False)
        get_conf_key = lambda conf: (sorted(conf.divisible_by_16), sorted(conf.equal_to_1), sorted(conf.ids_of_folded_args), sorted(conf.divisible_by_8))
        configs_key = [get_conf_key(conf) for conf in configs]
        env_vars_list = [f'{env_vars[k]}' for k in sorted(env_vars.keys())]
        key = f'{fn.cache_key}-{version_key}-{''.join(signature.values())}-{configs_key}-{constants}-{num_warps}-{num_stages}-{num_ctas}-{num_stages}-{enable_warp_specialization}-{enable_persistent}-{debug}-{target}-{env_vars_list}'
        return hashlib.md5(key.encode('utf-8')).hexdigest()
    assert isinstance(fn, str)
    ignore_version = kwargs.get('ignore_version', False)
    if ignore_version:
        return hashlib.md5(Path(fn).read_text().encode('utf-8')).hexdigest()
    return hashlib.md5((Path(fn).read_text() + version_key).encode('utf-8')).hexdigest()