import warnings
from contextlib import nullcontext
from typing import Any, Callable, List, Tuple, Union
from unittest.mock import patch
import torch
import torch.fx.traceback as fx_traceback
import torch.utils._pytree as pytree
from torch import Tensor
from torch._decomp.decompositions_for_rng import PhiloxStateTracker
from torch._guards import detect_fake_mode
from torch._prims_common import CUDARngStateHelper
from torch._subclasses.functional_tensor import FunctionalTensorMode
from torch.fx import Interpreter
from torch.fx.experimental.symbolic_shapes import definitely_false, sym_eq
from torch.nn.utils import stateless
from .. import config
from .collect_metadata_analysis import run_functionalized_fw_and_collect_metadata
from .functional_utils import (
from .logging_utils import setup_stacktrace_preservation_hooks
from .schemas import (
from .subclass_utils import (
from .utils import maybe_to_fresh_input
def create_functional_call(mod, params_spec, params_len):

    def functional_call(*args, **kwargs):
        with stateless._reparametrize_module(mod, pytree.tree_unflatten(args[:params_len], params_spec)):
            if isinstance(mod, torch.fx.GraphModule):
                with fx_traceback.preserve_node_meta(), warnings.catch_warnings():
                    warnings.filterwarnings('ignore', 'Anomaly Detection has been enabled.')
                    with torch.autograd.detect_anomaly(check_nan=False):
                        out = Interpreter(mod).run(*args[params_len:], **kwargs)
            else:
                out = mod(*args[params_len:], **kwargs)
        if not isinstance(out, (tuple, list)):
            raise RuntimeError('Graph output must be a tuple(). This is so that we can avoid pytree processing of the outputs. Please change the module to have tuple outputs or use aot_module instead.')
        return out
    return functional_call