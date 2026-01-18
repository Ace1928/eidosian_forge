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
def assert_ns_compare_dict_valid(self, act_compare_dict: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
    """
            Verifies that the act_compare_dict (output of Numeric Suite APIs) is valid:
            1. for each layer, results are recorded for two models
            2. number of seen tensors match
            3. shapes of each pair of seen tensors match
            """
    for layer_name, result_type_to_data in act_compare_dict.items():
        for result_type, layer_data in result_type_to_data.items():
            self.assertTrue(len(layer_data) == 2, f'Layer {layer_name} does not have exactly two model results.')
            model_name_0, model_name_1 = layer_data.keys()
            for res_idx in range(len(layer_data[model_name_0])):
                layer_data_0 = layer_data[model_name_0][res_idx]
                layer_data_1 = layer_data[model_name_1][res_idx]
                self.assertTrue(layer_data_0['type'] == layer_data_0['type'], f'Layer {layer_name}, {model_name_0} and {model_name_1} do not have the same type.')
                self.assertTrue(len(layer_data_0['values']) == len(layer_data_1['values']), f'Layer {layer_name}, {model_name_0} and {model_name_1} do not have the same number of seen Tensors.')
                is_weight_functional_conv1d = result_type == NSSingleResultValuesType.WEIGHT.value and ('conv1d' in layer_data_0['prev_node_target_type'] or 'conv1d' in layer_data_1['prev_node_target_type'])
                if not is_weight_functional_conv1d:
                    for idx in range(len(layer_data_0['values'])):
                        values_0 = layer_data_0['values'][idx]
                        values_1 = layer_data_1['values'][idx]
                        if isinstance(values_0, torch.Tensor):
                            self.assertTrue(values_0.shape == values_1.shape, f'Layer {layer_name}, {model_name_0} and {model_name_1} ' + f'have a shape mismatch at idx {idx}.')
                        elif isinstance(values_0, list):
                            values_0 = values_0[0]
                            values_1 = values_1[0]
                            self.assertTrue(values_0.shape == values_1.shape, f'Layer {layer_name}, {model_name_0} and {model_name_1} ' + f'have a shape mismatch at idx {idx}.')
                        else:
                            assert isinstance(values_0, tuple), f'unhandled type {type(values_0)}'
                            assert len(values_0) == 2
                            assert len(values_0[1]) == 2
                            assert values_0[0].shape == values_1[0].shape
                            assert values_0[1][0].shape == values_1[1][0].shape
                            assert values_0[1][1].shape == values_1[1][1].shape
                ref_node_name_0 = layer_data_0['ref_node_name']
                ref_node_name_1 = layer_data_1['ref_node_name']
                prev_node_name_0 = layer_data_0['prev_node_name']
                prev_node_name_1 = layer_data_1['prev_node_name']
                if layer_data_0['type'] == NSSingleResultValuesType.NODE_OUTPUT.value:
                    self.assertTrue(ref_node_name_0 == prev_node_name_0)
                    self.assertTrue(ref_node_name_1 == prev_node_name_1)
                elif layer_data_0['type'] == NSSingleResultValuesType.NODE_INPUT.value:
                    self.assertTrue(ref_node_name_0 != prev_node_name_0)
                    self.assertTrue(ref_node_name_1 != prev_node_name_1)