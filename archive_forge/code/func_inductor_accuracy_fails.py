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
def inductor_accuracy_fails(fx_g, args, check_str=None, *, require_fp64=False, ignore_non_fp=False):
    from torch._inductor.compile_fx import compile_fx_inner
    return backend_aot_accuracy_fails(fx_g, args, compile_fx_inner, require_fp64=require_fp64, ignore_non_fp=ignore_non_fp)