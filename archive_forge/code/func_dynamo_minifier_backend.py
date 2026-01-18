import argparse
import copy
import functools
import logging
import os
import shutil
import sys
import textwrap
from importlib import import_module
from typing import Union
import torch
import torch.fx as fx
from torch._dynamo.debug_utils import (
from torch.fx.experimental.symbolic_shapes import fx_placeholder_targets
from torch.hub import tqdm
from .. import config
from ..backends.registry import lookup_backend, register_debug_backend
from ..debug_utils import clone_inputs_retaining_gradness
from math import inf
import torch
from torch import tensor, device
import torch.fx as fx
import torch._dynamo
from torch._dynamo.testing import rand_strided
from torch._dynamo.debug_utils import run_fwd_maybe_bwd
@register_debug_backend
def dynamo_minifier_backend(gm, example_inputs, compiler_name):
    from functorch.compile import minifier
    compiler_fn = lookup_backend(compiler_name)
    example_inputs = [i.node.hint if isinstance(i, torch.SymInt) else i for i in example_inputs]
    try:
        compiled_gm = compiler_fn(gm, example_inputs)
        run_fwd_maybe_bwd(compiled_gm, example_inputs)
        raise ValueError('No issue was detected')
    except Exception as exc:
        orig_failure = str(exc)
        log.warning('Compiled Fx GraphModule failed. Creating script to minify the error.')
        dump_state_fn = functools.partial(dump_backend_state, compiler_name=compiler_name)
        dump_state_fn(fx.GraphModule(gm, copy.deepcopy(gm.graph)), example_inputs)
        fails_fn = functools.partial(backend_fails, compiler_fn=compiler_fn, orig_failure=orig_failure)
        minifier(gm, example_inputs, module_fails=fails_fn, dump_state=dump_state_fn)
    return gm