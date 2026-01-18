import itertools
from contextlib import nullcontext
from functools import partial, wraps
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import patch
import torch
import torch.nn as nn
import torch.utils._pytree as pytree
import torch.utils.dlpack
from torch import Tensor
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo import compiled_autograd
from torch._dynamo.utils import dynamo_timed, preserve_rng_state
from torch._guards import detect_fake_mode
from torch._subclasses import FakeTensor, FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import (
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch._decomp.decompositions_for_rng import PhiloxStateTracker, rng_decompositions
from . import config
from .partitioners import default_partition
from ._aot_autograd.utils import (  # noqa: F401
from ._aot_autograd.logging_utils import (  # noqa: F401
from ._aot_autograd.functional_utils import (  # noqa: F401
from ._aot_autograd.schemas import (  # noqa: F401
from ._aot_autograd.subclass_utils import (  # noqa: F401
from ._aot_autograd.collect_metadata_analysis import (  # noqa: F401
from ._aot_autograd.input_output_analysis import (  # noqa: F401
from ._aot_autograd.traced_function_transforms import (  # noqa: F401
from ._aot_autograd.runtime_wrappers import (  # noqa: F401
from ._aot_autograd.dispatch_and_compile_graph import (  # noqa: F401
from ._aot_autograd.jit_compile_runtime_wrappers import (  # noqa: F401
def flattened_joint(*args):
    fake_tangents = [None for _ in range(metadata.num_outputs + metadata.num_mutated_inp_runtime_indices)]
    fw_outs, gradients = fx_g(args, fake_tangents)
    assert len(gradients) == len(args)
    output_gradients = []
    for i, (a, grad) in enumerate(zip(args, gradients)):
        if isinstance(a, torch.Tensor) and a.requires_grad:
            assert grad is not None, 'Found a parameter that did not receive a gradient.\n"This is most likely a bug, but if this needs to be supported please comment on this Github issue:\nhttps://github.com/pytorch/pytorch/issues/101192\n'
            output_gradients.append(grad)
        else:
            assert grad is None
    return (*fw_outs, *output_gradients)