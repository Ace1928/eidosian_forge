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
def generate_config_string(*, stable_output=False):
    import torch._functorch.config
    import torch._inductor.config
    if stable_output:
        return '# config omitted due to stable_output=True'
    return f'import torch._dynamo.config\nimport torch._inductor.config\nimport torch._functorch.config\nimport torch.fx.experimental._config\n{torch._dynamo.config.codegen_config()}\n{torch._inductor.config.codegen_config()}\n{torch._functorch.config.codegen_config()}\n{torch.fx.experimental._config.codegen_config()}\n'