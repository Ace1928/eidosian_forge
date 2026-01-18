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
def checkObservers(self, module, propagate_qconfig_list=None, prepare_custom_config_dict=None):
    """Checks the module or module's leaf descendants
            have observers in preparation for quantization
        """
    if propagate_qconfig_list is None:
        propagate_qconfig_list = get_default_qconfig_propagation_list()
    if prepare_custom_config_dict is None:
        prepare_custom_config_dict = {}
    float_to_observed_module_class_mapping = prepare_custom_config_dict.get('float_to_observed_custom_module_class', {})

    def is_leaf_module(module):
        submodule_name_count = 0
        for name, _ in module.named_children():
            if name != 'activation_post_process':
                submodule_name_count += 1
        return submodule_name_count == 0
    if hasattr(module, 'qconfig') and module.qconfig is not None and (is_leaf_module(module) and (not isinstance(module, torch.nn.Sequential)) and (type(module) in propagate_qconfig_list) or type(module) in float_to_observed_module_class_mapping.keys()) and (not isinstance(module, torch.ao.quantization.DeQuantStub)):
        self.assertTrue(hasattr(module, 'activation_post_process'), 'module: ' + str(type(module)) + ' do not have observer')
    if type(module) not in get_default_qat_module_mappings().values() and type(module) not in float_to_observed_module_class_mapping.values() and (not isinstance(module, _FusedModule)):
        for child in module.children():
            if type(child) in [nn.Dropout]:
                continue
            self.checkObservers(child, propagate_qconfig_list, prepare_custom_config_dict)