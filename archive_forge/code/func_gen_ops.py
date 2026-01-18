import functools
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, List, Optional
import sympy
import torch
from ...codecache import cache_dir
from ...config import cuda as inductor_cuda_config
from ...ir import Layout
from .cuda_env import get_cuda_arch, get_cuda_version
def gen_ops() -> List[Any]:
    """
    Generates all supported CUTLASS operations.
    """
    arch = get_cuda_arch()
    version = get_cuda_version()
    return _gen_ops_cached(arch, version)