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
def script_module(*args, **kwargs):
    formals, tensors, actuals = get_script_args(args)
    method_args = ', '.join(['self'] + actuals)
    call_args_str = ', '.join(actuals)
    call = f'self.submodule({call_args_str})'
    script = script_method_template.format(method_args, call)
    submodule_constants = []
    if kwargs.get('is_constant'):
        submodule_constants = ['submodule']

    class TheModule(torch.jit.ScriptModule):
        __constants__ = submodule_constants

        def __init__(self):
            super().__init__()
            self.submodule = nn_module(*constructor_args)

    def make_module(script):
        module = TheModule()
        str(module)
        module.define(script)
        return module
    module = make_module(script)
    if self:
        self.assertExportImportModule(module, tensors)
        module(*args)
    create_script_module.last_graph = module.graph
    return module