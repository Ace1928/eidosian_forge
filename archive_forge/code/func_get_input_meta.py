import copy
import logging
import os
import pickle
import random
from contextlib import contextmanager
from functools import partial
from typing import Callable, Union
import sympy
import torch
from torch import SymInt
import torch.fx as fx
import torch.nn as nn
from torch._decomp import get_decompositions
from torch.fx.experimental.symbolic_shapes import bind_symbols
from .aot_autograd import aot_function, aot_module, make_boxed_compiler
from .compile_utils import strip_overloads
from .partitioners import (
import torch.utils._pytree as pytree
import torch
import torch.fx as fx
from functorch.compile import minifier, check_nvfuser_subprocess, check_nvfuser_correctness_subprocess
from foo import FxModule
def get_input_meta(args):
    input_meta = []
    if len(args) > 0 and isinstance(args[0], tuple):
        input_meta += get_input_meta(args[0])
        input_meta += get_input_meta(args[1])
        return input_meta
    for arg in args:
        if type(arg) == int or type(arg) == float:
            input_meta.append((type(arg),))
        else:
            input_meta.append((type(arg), arg.shape, arg.stride(), arg.dtype, arg.device))
    return input_meta