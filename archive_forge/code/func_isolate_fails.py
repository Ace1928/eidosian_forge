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
def isolate_fails(fx_g, args, compiler_name: str, env=None, save_dir=None, accuracy=None, tracing_mode=None, check_str=None):
    if env is None:
        env = {}
    subdir = os.path.join(os.getcwd(), 'isolate')
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)
    file_name = os.path.join(subdir, f'{str(uuid.uuid4())[:5]}.py')
    with open(file_name, 'w') as fd:
        save_graph_repro(fd, fx_g, args, compiler_name, save_dir=save_dir, command='minifier-query', accuracy=accuracy, tracing_mode=tracing_mode, check_str=check_str)
    new_env = os.environ.copy()
    new_env = {**new_env, **env}
    stdout, stderr = (TemporaryFile(), TemporaryFile())
    if use_buck:
        cmd = BuckTargetWriter(file_name).write(print_msg=False)
    else:
        cmd = ['python', file_name]
    p = subprocess.Popen(cmd, cwd=subdir, stdout=stdout, stderr=stderr, env=new_env)
    p.wait()
    stdout.seek(0)
    stderr.seek(0)
    print(textwrap.indent(stdout.read().decode('utf-8'), prefix='>>  '), file=sys.stdout)
    print(textwrap.indent(stderr.read().decode('utf-8'), prefix='>>  '), file=sys.stderr)
    return p.returncode != 0