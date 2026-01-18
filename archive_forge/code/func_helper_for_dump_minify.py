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
def helper_for_dump_minify(contents):
    minified_repro_path = get_minifier_repro_path()
    log.warning('Writing minified repro to:\n%s', minified_repro_path)
    if use_buck:
        BuckTargetWriter(minified_repro_path).write()
    try:
        with open(minified_repro_path, 'w') as fd:
            fd.write(contents)
    except OSError as e:
        log.exception(e)
        raise NotImplementedError('Could not write to {minified_repro_path}') from e