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
def save_graph_repro(fd, gm, args, compiler_name, *, stable_output=False, save_dir=None, command='run', accuracy=None, tracing_mode=None, check_str=None):
    fd.write(generate_compiler_repro_string(gm, args, stable_output=stable_output, save_dir=save_dir))
    if accuracy is None:
        accuracy = '_accuracy' in compiler_name
    if tracing_mode is None:
        tracing_mode = 'real'
        if any((has_free_symbols(a) for a in args)):
            tracing_mode = 'symbolic'
    fd.write("if __name__ == '__main__':\n")
    fd.write('    from torch._dynamo.repro.after_aot import run_repro\n')
    fd.write(f'    with torch.no_grad():        run_repro(mod, load_args, accuracy={accuracy!r}, command={command!r}, save_dir={save_dir!r}, tracing_mode={tracing_mode!r}, check_str={check_str!r})\n')