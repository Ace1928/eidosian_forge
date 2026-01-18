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
def clone_inputs_retaining_gradness(example_inputs):
    """
    This clone inputs is different from utils clone_input. In case of minifier,
    all the tensors are leaf tensors while creating a new graph. So, we set the
    requires_grad field w/o checking the leafness of the tensor.
    """
    cloned_inputs = clone_inputs(example_inputs)
    for idx in range(len(example_inputs)):
        if isinstance(cloned_inputs[idx], torch.Tensor):
            cloned_inputs[idx].requires_grad_(example_inputs[idx].requires_grad)
    return cloned_inputs