import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.nn.intrinsic.quantized.dynamic as nniqd
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
from torch.ao.nn.intrinsic import _FusedModule
import torch.distributed as dist
from torch.testing._internal.common_utils import TestCase, TEST_WITH_ROCM
from torch.ao.quantization import (
from torch.ao.quantization import QuantWrapper, QuantStub, DeQuantStub, \
from torch.ao.quantization.quantization_mappings import (
from torch.testing._internal.common_quantized import (
from torch.jit.mobile import _load_for_lite_interpreter
import copy
import io
import functools
import time
import os
import unittest
import numpy as np
from torch.testing import FileCheck
from typing import Callable, Tuple, Dict, Any, Union, Type, Optional
import torch._dynamo as torchdynamo
def printGraphModule(self, graph_module, print_str=True):
    modules = dict(graph_module.named_modules(remove_duplicate=False))
    node_infos = []
    for n in graph_module.graph.nodes:
        node_info = ' '.join(map(repr, [n.op, n.name, n.target, n.args, n.kwargs]))
        if n.op == 'call_module':
            node_info += ' module type: ' + repr(type(modules[n.target]))
        node_infos.append(node_info)
    str_to_print = '\n'.join(node_infos)
    if print_str:
        print(str_to_print)
    return str_to_print