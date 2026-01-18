import argparse
import copy
import functools
import io
import logging
import os
import shutil
import subprocess
import sys
import textwrap
import uuid
from importlib import import_module
from tempfile import TemporaryFile
from typing import Any, Callable, Dict, Union
import torch
import torch.fx as fx
import torch.nn as nn
from torch._dynamo.debug_utils import (
from torch._dynamo.utils import clone_inputs, counters, same
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import (
from torch.hub import tqdm
from .. import config
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims
def inner_debug_fn(real_inputs):
    """
            Aot Autograd fw_compiler and bw_compiler can have fake tensors. So,
            example_inputs can be fake tensors. We can call compiler_fn (which is
            inductor or nvfuser) with fake tensors but the actually compiled_fn
            should be called with real tensors. Therefore, the actual invocation
            is deferred.
            """
    fake_mode = FakeTensorMode()
    copy_tensor_attrs = [fake_mode.from_tensor(x) if isinstance(x, torch.Tensor) else x for x in real_inputs]
    if config.repro_level == 3:
        dump_to_minify(fx.GraphModule(gm, orig_graph), real_inputs, compiler_name)
    if config.repro_level == 4:
        if compiler_name != 'inductor':
            raise NotImplementedError('Accuracy minification is supported for inductor only')
        if backend_aot_accuracy_fails(gm, real_inputs, compiler_fn):
            log.warning('Accuracy failed for the AOT Autograd graph %s', graph_name)
            dump_compiler_graph_state(fx.GraphModule(gm, orig_graph), real_inputs, f'{compiler_name}_accuracy')
            dump_to_minify(fx.GraphModule(gm, orig_graph), real_inputs, f'{compiler_name}_accuracy')
            raise AccuracyError('Bad accuracy detected')
        else:
            return inner_compiled_fn(real_inputs)
    else:
        try:
            out = inner_compiled_fn(real_inputs)
            for arg in example_inputs:
                if isinstance(arg, torch.Tensor) and arg.is_cuda:
                    torch.cuda.synchronize()
                    break
            return out
        except Exception as e:
            if config.repro_level == 1:
                dump_compiler_graph_state(fx.GraphModule(gm, orig_graph), copy_tensor_attrs, compiler_name)
            elif config.repro_level == 2:
                dump_to_minify(fx.GraphModule(gm, orig_graph), copy_tensor_attrs, compiler_name)
            raise