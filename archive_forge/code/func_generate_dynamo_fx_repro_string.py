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
def generate_dynamo_fx_repro_string(gm, args, compiler_name, check_accuracy=False, *, stable_output=False, save_dir=None, command='run'):
    """
    Generate a repro string for backend-agnostic minified version.
    """
    model_str = NNModuleToString.convert(gm)
    writer = InputWriter(save_dir, stable_hash=True)
    for placeholder, arg in zip(fx_placeholder_targets(gm), args):
        if isinstance(arg, (int, torch.SymInt)):
            writer.symint(placeholder, arg)
        elif isinstance(arg, torch.Tensor):
            writer.tensor(placeholder, arg)
        else:
            raise TypeError(f'arg is neither SymInt/int nor torch.Tensor, {arg}')
    load_args = '\n'.join(writer.lines())
    return textwrap.dedent(f"\nfrom math import inf\nimport torch\nfrom torch import tensor, device\nimport torch.fx as fx\nimport torch._dynamo\nfrom torch._dynamo.testing import rand_strided\nfrom torch._dynamo.debug_utils import run_fwd_maybe_bwd\n\n{generate_config_string(stable_output=stable_output)}\n\n{extra_imports}\n\n{model_str}\nmod = Repro()\n\n{load_args}\n\nif __name__ == '__main__':\n    from torch._dynamo.repro.after_dynamo import run_repro\n    run_repro(mod, load_args, accuracy={check_accuracy!r}, command={command!r},\n        save_dir={save_dir!r}, autocast={torch.is_autocast_enabled()!r}, backend={compiler_name!r})\n")