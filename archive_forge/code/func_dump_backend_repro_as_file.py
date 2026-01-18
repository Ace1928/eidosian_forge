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
def dump_backend_repro_as_file(gm, args, compiler_name, check_accuracy=False):
    """
    Saves the repro to a repro.py file
    """
    curdir = os.getcwd()
    subdir = os.path.join(os.getcwd(), 'checkpoints')
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)
    file_name = os.path.join(subdir, f'minified_{len(gm.graph.nodes)}_nodes.py')
    log.warning('Writing checkpoint with %s nodes to %s', len(gm.graph.nodes), file_name)
    with open(file_name, 'w') as fd:
        fd.write(generate_dynamo_fx_repro_string(gm, args, compiler_name, check_accuracy, save_dir=subdir))
    latest_repro = os.path.join(curdir, 'repro.py')
    log.warning('Copying %s to %s for convenience', file_name, latest_repro)
    if use_buck:
        BuckTargetWriter(latest_repro).write()
    shutil.copyfile(file_name, latest_repro)