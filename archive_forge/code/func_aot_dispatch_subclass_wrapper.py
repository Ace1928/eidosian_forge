import collections
import pprint
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.utils.dlpack
from torch import Tensor
from torch._guards import DuplicateInputs, TracingContext
from torch._prims_common import CUDARngStateHelper
from torch.multiprocessing.reductions import StorageWeakRef
from .. import config
from .collect_metadata_analysis import run_functionalized_fw_and_collect_metadata
from .functional_utils import gen_alias_from_base
from .input_output_analysis import (
from .logging_utils import describe_input, format_guard_bug_msg
from .schemas import (
from .subclass_utils import (
from .utils import (
def aot_dispatch_subclass_wrapper(runtime_fn: Callable, *, subclass_metas: List[Union[int, SubclassCreationMeta]], num_fw_outs_saved_for_bw: Optional[int]) -> Callable:

    def inner_fn(args):
        unwrapped_args = unwrap_tensor_subclasses(args, is_joint_structure=False)
        unwrapped_outs = runtime_fn(unwrapped_args)
        wrapped_outs = wrap_tensor_subclasses(unwrapped_outs, subclass_metas=subclass_metas, num_fw_outs_saved_for_bw=num_fw_outs_saved_for_bw, is_runtime=True)
        return wrapped_outs
    inner_fn._boxed_call = True
    return inner_fn