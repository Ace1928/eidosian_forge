import copy
import functools
import getpass
import itertools
import logging
import os
import subprocess
import tempfile
import textwrap
from collections import Counter
from importlib import import_module
from typing import Callable, Optional, TypeVar
import torch
import torch._prims_common as utils
import torch._subclasses.meta_utils
from torch._dynamo.testing import rand_strided
from torch._prims_common import is_float_dtype
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._content_store import ContentStoreReader, ContentStoreWriter
from . import config
from .utils import clone_inputs, get_debug_dir
import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
def cast_dtype_args_to_fp64(model):
    for node in model.graph.nodes:
        if node.op == 'call_function' and node.target == torch.ops.prims.convert_element_type.default:
            assert len(node.args) == 2
            if is_float_dtype(node.args[1]) and node.args[1] != torch.float64:
                node.args = (node.args[0], torch.float64)
        if node.op == 'call_function':
            dtype = node.kwargs.get('dtype')
            if dtype is not None and is_float_dtype(dtype):
                new_kwargs = dict(node.kwargs)
                new_kwargs['dtype'] = torch.float64
                node.kwargs = new_kwargs
    model.graph.lint()
    model.recompile()
    return model