import os
import re
from enum import Enum
from inspect import signature
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from packaging import version
from transformers.utils import logging
import onnxruntime as ort
from ..exporters.onnx import OnnxConfig, OnnxConfigWithLoss
from ..utils.import_utils import _is_package_available
def get_ordered_input_names(input_names: List[str], func: Callable) -> List[str]:
    """
    Returns the input names from input_names keys ordered according to the signature of func. This is especially useful with the
    forward function when using IO Binding, as the input order of the ONNX and forward may be different.

    Method inspired from OnnxConfig.ordered_inputs.

    Args:
        input_names (`List[str]`):
            Names of the inputs of the ONNX model.
        func (`Callable`):
            Callable to remap the input_names order to.

    """
    signature_func = signature(func)
    _ordered_input_names = []
    for param in signature_func.parameters:
        param_regex = re.compile(f'{param}(\\.\\d*)?')
        for name in input_names:
            if re.search(param_regex, name):
                _ordered_input_names.append(name)
    return _ordered_input_names