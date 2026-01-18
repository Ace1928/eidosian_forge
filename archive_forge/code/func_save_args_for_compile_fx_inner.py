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
def save_args_for_compile_fx_inner(*args, **kwargs):
    """
    This function is used to save arguments for a compile_fx_inner function call
    to the file system.  Later on one can replay the compile_fx_inner call
    with the saved arguments using load_args_and_run_compile_fx_inner.
    """
    folder = '/tmp/inductor_saved_args'
    if not os.path.exists(folder):
        os.mkdir(folder)

    def handle_tensor(x):
        """
        Pickle FakeTensor will result in error:
        AttributeError: Can't pickle local object 'WeakValueDictionary.__init__.<locals>.remove'

        Convert all Tensor to metadata. This may also makes pickle faster.
        """
        if isinstance(x, torch.Tensor):
            return TensorMetadataHolder(_extract_tensor_metadata(x), x.device)
        else:
            return x
    args_to_save, kwargs_to_save = tree_map(handle_tensor, (args, kwargs))
    fn_name = 'compile_fx_inner'
    path = f'{folder}/{fn_name}_{next(save_args_cnt)}.pkl'
    with open(path, 'wb') as f:
        pickle.dump((args_to_save, kwargs_to_save), f)
    if log.isEnabledFor(logging.DEBUG):
        message = f'\nArguments for a compile_fx_inner call is saved to {path}. To replay the call,\nrun the following:\n\nfrom torch._inductor.debug import load_args_and_run_compile_fx_inner\nload_args_and_run_compile_fx_inner({path!r})\n        '
        print(message)