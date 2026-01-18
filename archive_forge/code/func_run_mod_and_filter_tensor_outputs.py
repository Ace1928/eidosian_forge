import contextlib
import copy
import functools
import inspect
import os
import re
import warnings
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar
from typing_extensions import ParamSpec
import torch
from torch._jit_internal import (
from torch.autograd import function
from torch.jit._script import _CachedForward, script, ScriptModule
from torch.jit._state import _enabled, _python_cu
from torch.nn import Module
from torch.testing._comparison import default_tolerances
def run_mod_and_filter_tensor_outputs(mod, inputs, running_what):
    try:
        if isinstance(inputs, dict) and example_inputs_is_kwarg:
            outs = wrap_retval(mod(**inputs))
        else:
            outs = wrap_retval(mod(*_clone_inputs(inputs)))
        outs = [out for out in outs if isinstance(out, torch.Tensor)]
        return outs
    except Exception as e:
        graph_diff_errors, tensor_compare_errors = graph_diagnostic_info()
        msg = f'encountered an exception while running the {running_what} with test inputs.\nException:\n{indent(str(e))}'
        raise TracingCheckError(graph_diff_errors, tensor_compare_errors, extra_msg=msg) from e