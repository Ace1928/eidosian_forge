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
def log_results(name, input_nodes, timings, elapse):
    if not (config.max_autotune or config.max_autotune_gemm) or not PRINT_AUTOTUNE:
        return
    sizes = ', '.join(['x'.join(map(str, V.graph.sizevars.size_hints(n.get_size(), fallback=config.unbacked_symint_fallback))) for n in input_nodes])
    n = None if log.getEffectiveLevel() == logging.DEBUG else 10
    top_k = sorted(timings, key=timings.__getitem__)[:n]
    best = top_k[0]
    best_time = timings[best]
    sys.stderr.write(f'AUTOTUNE {name}({sizes})\n')
    for choice in top_k:
        result = timings[choice]
        if result:
            sys.stderr.write(f'  {choice.name} {result:.4f} ms {best_time / result:.1%}\n')
        else:
            sys.stderr.write(f'  {choice.name} {result:.4f} ms <DIVIDED BY ZERO ERROR>\n')
    autotune_type_str = 'SubProcess' if config.autotune_in_subproc else 'SingleProcess'
    sys.stderr.write(f'{autotune_type_str} AUTOTUNE takes {elapse:.4f} seconds\n')