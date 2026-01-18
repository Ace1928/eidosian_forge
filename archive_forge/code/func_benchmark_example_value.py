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
def benchmark_example_value(node):
    """
        Convert an ir.Buffer into a concrete torch.Tensor we can use for
        benchmarking.
        """
    if isinstance(node, ir.Layout):
        node = ir.Buffer('fake', node)
    if isinstance(node, ir.BaseView):
        node = node.unwrap_view()
    with preserve_rng_state():
        return rand_strided(V.graph.sizevars.size_hints(node.get_size(), fallback=config.unbacked_symint_fallback), V.graph.sizevars.size_hints(node.get_stride(), fallback=config.unbacked_symint_fallback), device=node.get_device(), dtype=node.get_dtype(), extra_size=node.layout.offset)