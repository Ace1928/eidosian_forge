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
def inductor_fails(fx_g, args, check_str=None):
    has_cuda = False
    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.is_cuda:
            has_cuda = True
            break

    def sync():
        if has_cuda:
            torch.cuda.synchronize()
    from torch._inductor.compile_fx import compile_fx_inner
    try:
        result = fx_g(*args)
        assert isinstance(result, (tuple, list))
        assert not any((isinstance(x, (tuple, list)) for x in result))
    except Exception:
        return False
    sync()
    try:
        compile_mod = compile_fx_inner(fx_g, args)
        compile_mod(args)
        sync()
    except Exception as e:
        if check_str is not None and check_str not in repr(e):
            return False
        print(repr(e))
        return True
    return False