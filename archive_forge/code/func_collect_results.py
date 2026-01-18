import contextlib
import dis
import functools
import logging
import os.path
import random
import re
import sys
import types
import unittest
from typing import List, Optional, Sequence, Union
from unittest.mock import patch
import torch
from torch import fx
from torch._dynamo.output_graph import OutputGraph
from . import config, eval_frame, optimize_assert, reset
from .bytecode_transformation import (
from .guards import CheckFunctionManager, GuardedCode
from .utils import same
def collect_results(model, prediction, loss, example_inputs):
    results = []
    results.append(prediction)
    results.append(loss)
    grads = dict()
    params = dict()
    for name, param in model.named_parameters():
        if isinstance(model, eval_frame.OptimizedModule):
            name = remove_optimized_module_prefix(name)
        param_copy = param
        grad = param.grad
        if param.grad is None:
            grad = torch.zeros_like(param)
        grads[name + '.grad'] = grad
        params[name] = param_copy
    results.append(grads)
    results.append(params)
    buffers = dict()
    for name, buffer in model.named_buffers():
        if isinstance(model, eval_frame.OptimizedModule):
            name = remove_optimized_module_prefix(name)
        buffers[name] = buffer
    results.append(buffers)
    for example in example_inputs:
        if isinstance(example, (tuple, list)):
            for inp in example:
                if isinstance(inp, torch.Tensor):
                    results.append(inp.grad)
        elif isinstance(example, torch.Tensor):
            results.append(example.grad)
    return results