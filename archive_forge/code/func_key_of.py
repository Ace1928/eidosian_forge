import builtins
import functools
import inspect
import itertools
import logging
import sys
import textwrap
import time
from io import StringIO
from typing import Any, Callable, Dict, List, Optional, Type, Union
from unittest.mock import patch
import sympy
import torch
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import counters, identity, preserve_rng_state
from . import config, ir
from .autotune_process import TensorMeta, TritonBenchmarkRequest
from .codecache import code_hash, PersistentCache, PyCodeCache
from .codegen.common import ChoiceCaller, IndentedBuffer, KernelTemplate
from .codegen.triton import texpr, TritonKernel, TritonPrinter, TritonScheduling
from .codegen.triton_utils import config_of, signature_to_meta
from .exc import CUDACompileError
from .utils import do_bench, Placeholder, sympy_dot, sympy_product, unique
from .virtualized import V
from . import lowering  # noqa: F401
@staticmethod
def key_of(node):
    """
        Extract the pieces of an ir.Buffer that we should invalidate cached
        autotuning results on.
        """
    sizevars = V.graph.sizevars
    return (node.get_device().type, str(node.get_dtype()), *sizevars.size_hints(node.get_size(), fallback=config.unbacked_symint_fallback), *sizevars.size_hints(node.get_stride(), fallback=config.unbacked_symint_fallback), sizevars.size_hint(node.get_layout().offset, fallback=config.unbacked_symint_fallback))