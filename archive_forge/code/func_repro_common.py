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
def repro_common(options, mod, load_args):
    assert not any(mod.named_parameters())
    for n, b in mod.named_buffers():
        if b.numel() > MAX_CONSTANT_NUMEL_INLINE:
            log.warning('Constant %s was not serialized, generated random data instead. If you think this is affecting you, please comment on https://github.com/pytorch/pytorch/issues/100468', n)
    if not hasattr(load_args, '_version'):
        log.warning('load_args does not have a _version attribute, please file a bug to PyTorch and describe how you generate this repro script')
    elif load_args._version > 0:
        log.warning('load_args is version %s, but this version of PyTorch only supports version 0.  We will try to run it anyway but there may be an incompatibility; if so, try upgrading your version of PyTorch.', load_args._version)
    nop_reader = NopInputReader()
    load_args(nop_reader)
    with tqdm(desc='Loading inputs', total=nop_reader.total) as pbar:
        input_reader = InputReader(save_dir=options.save_dir, pbar=pbar)
        load_args(input_reader)
        args = input_reader.args
    mod = make_fx(mod, tracing_mode=options.tracing_mode)(*args)
    torch._inductor.config.generate_intermediate_hooks = True
    return (mod, args)