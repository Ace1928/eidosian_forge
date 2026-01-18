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
def backend_fails(gm, example_inputs, compiler_fn, orig_failure):
    """
    Minifier uses this function to identify if the minified graph module fails
    with the same error.

    One caveat is that minifier can potentially go into a wrong direction when
    the resulting graph module fails for a different reason. To avoid this, we
    save the string for the original exception and check similarity between new
    and old exception. They can be somewhat different in some cases, when the
    exception string depends on the failing node information. So, we have a
    loose similarity metric to guide the minifier path.
    """
    from difflib import SequenceMatcher
    try:
        run_fwd_maybe_bwd(gm, clone_inputs_retaining_gradness(example_inputs))
        compiled_gm = compiler_fn(gm, example_inputs)
        run_fwd_maybe_bwd(compiled_gm, clone_inputs_retaining_gradness(example_inputs))
        return False
    except Exception as e:
        new_failure = str(e)
        if SequenceMatcher(None, orig_failure, new_failure).ratio() > 0.5:
            return True
        return False