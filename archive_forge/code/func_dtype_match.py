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
def dtype_match(torch_dtype: Optional[torch.dtype], cutlass_dtype: 'cutlass_library.library.DataType') -> bool:
    assert try_import_cutlass()
    import cutlass_library
    if torch_dtype == torch.float:
        return cutlass_dtype == cutlass_library.library.DataType.f32 or cutlass_dtype == cutlass_library.library.DataType.tf32
    elif torch_dtype == torch.half:
        return cutlass_dtype == cutlass_library.library.DataType.f16
    elif torch_dtype == torch.bfloat16:
        return cutlass_dtype == cutlass_library.library.DataType.bf16
    else:
        return False