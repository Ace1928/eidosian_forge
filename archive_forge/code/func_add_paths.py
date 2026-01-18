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
def add_paths(exc):
    exc.minifier_path = os.path.join(minifier_dir(), 'minifier_launcher.py')
    if use_buck:
        exc.buck_command = ' '.join(BUCK_CMD_PREFIX + [BuckTargetWriter(exc.minifier_path).cmd_line_path])