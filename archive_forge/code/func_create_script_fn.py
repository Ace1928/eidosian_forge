from torch.jit.annotations import BroadcastingList2, BroadcastingList3  # noqa: F401
import torch.nn.functional as F
import torch
import torch.cuda
import torch.jit
import torch.jit._logging
import torch.jit.frontend
from torch.testing._internal.common_nn import module_tests, new_module_tests
from torch.testing._internal.common_utils import is_iterable_of_tensors
import collections
from copy import deepcopy
from typing import Any, Dict, List, Union
import math  # noqa: F401
from torch import inf
def create_script_fn(self, method_name, func_type):

    def script_fn(*args, **kwargs):
        fn, tensors = gen_script_fn_and_args(method_name, func_type, *args, **kwargs)
        self.assertExportImport(fn.graph, tensors)
        output = fn(*tensors)
        script_fn.last_graph = fn.graph_for(*tensors)
        return output
    return script_fn