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
def checkGraphModeOp(self, module, inputs, quantized_op, tracing=False, debug=False, check=True, eval_mode=True, dynamic=False, qconfig=None):
    if debug:
        print('Testing:', str(module))
    qconfig_dict = {'': get_default_qconfig(torch.backends.quantized.engine)}
    if eval_mode:
        module = module.eval()
    if dynamic:
        qconfig_dict = {'': default_dynamic_qconfig if qconfig is None else qconfig}
    model = get_script_module(module, tracing, inputs[0]).eval()
    if debug:
        print('input graph:', model.graph)
    models = {}
    outputs = {}
    for debug in [True, False]:
        if dynamic:
            models[debug] = quantize_dynamic_jit(model, qconfig_dict, debug=debug)
            outputs[debug] = models[debug](inputs)
        else:
            inputs_copy = copy.deepcopy(inputs)
            models[debug] = quantize_jit(model, qconfig_dict, test_only_eval_fn, [inputs_copy], inplace=False, debug=debug)
            outputs[debug] = models[debug](*inputs[0])
    if debug:
        print('debug graph:', models[True].graph)
        print('non debug graph:', models[False].graph)
    if check:
        self.assertEqual(outputs[True], outputs[False])
        FileCheck().check(quantized_op).run(models[False].graph)
    return models[False]