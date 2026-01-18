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
def same_two_models(gm, opt_gm, example_inputs, only_fwd=False, *, require_fp64=False, ignore_non_fp=False):
    """
    Check two models have same accuracy.

    require_fp64: if True, raise an error if we unable to calculate the fp64 reference
    ignore_non_fp: if True, do not compare outputs which are not floating point.  This
        is mostly useful for the minifier (which wants to avoid quantizing floating point
        error into integer/boolean error)
    """
    from .eval_frame import OptimizedModule
    from .testing import named_buffers_for_optimized_module, named_parameters_for_optimized_module
    from .utils import same
    if isinstance(gm, OptimizedModule):
        gm.named_parameters = named_parameters_for_optimized_module(gm)
        gm.named_buffers = named_buffers_for_optimized_module(gm)
    if isinstance(opt_gm, OptimizedModule):
        opt_gm.named_parameters = named_parameters_for_optimized_module(opt_gm)
        opt_gm.named_buffers = named_buffers_for_optimized_module(opt_gm)
    ref = run_fwd_maybe_bwd(gm, example_inputs, only_fwd)
    fp64_ref = None
    if config.same_two_models_use_fp64:
        try:
            fp64_model, fp64_examples = cast_to_fp64(copy.deepcopy(gm), clone_inputs_retaining_gradness(example_inputs))
            fp64_ref = run_fwd_maybe_bwd(fp64_model, fp64_examples, only_fwd)
        except Exception:
            if require_fp64:
                raise RuntimeError('Could not generate fp64 outputs')
            log.warning('Could not generate fp64 outputs')
    try:
        res = run_fwd_maybe_bwd(opt_gm, example_inputs, only_fwd)
    except Exception as e:
        log.exception('While minifying the program in accuracy minification mode, ran into a runtime exception which is likely an unrelated issue. Skipping this graph.')
        return True
    passing = same(ref, res, fp64_ref, tol=config.repro_tolerance, equal_nan=True, ignore_non_fp=ignore_non_fp)
    return passing