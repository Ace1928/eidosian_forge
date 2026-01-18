import collections
import contextlib
import cProfile
import dataclasses
import functools
import itertools
import logging
import os
import os.path
import pickle
import pstats
import shutil
import subprocess
from typing import Any, Dict, List, Optional
from unittest.mock import patch
from functorch.compile import draw_graph, get_aot_graph_name, get_graph_being_compiled
import torch
from torch import fx as fx
from torch._dynamo.repro.after_aot import save_graph_repro, wrap_compiler_debug
from torch._dynamo.utils import get_debug_dir
from torch.fx.graph_module import GraphModule
from torch.fx.passes.shape_prop import _extract_tensor_metadata, TensorMetadata
from torch.fx.passes.tools_common import legalize_graph
from torch.utils._pytree import tree_map
from . import config, ir  # noqa: F811, this is needed
from .scheduler import (
from .virtualized import V
from torch._inductor.debug import load_args_and_run_compile_fx_inner
def load_args_and_run_compile_fx_inner(path: str):
    from torch._inductor.compile_fx import compile_fx_inner
    with open(path, 'rb') as f:
        args, kwargs = pickle.load(f)

    def handle_tensor(x):
        if isinstance(x, TensorMetadataHolder):
            return torch._dynamo.testing.rand_strided(x.tensor_metadata.shape, x.tensor_metadata.stride, x.tensor_metadata.dtype, x.device)
        else:
            return x
    fake_mode = torch._subclasses.FakeTensorMode(allow_non_fake_inputs=True)
    with fake_mode, config.patch('save_args', False):
        args, kwargs = tree_map(handle_tensor, (args, kwargs))
        return compile_fx_inner(*args, **kwargs)