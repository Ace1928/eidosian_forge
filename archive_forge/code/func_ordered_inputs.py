import copy
import enum
import gc
import inspect
import itertools
import os
import re
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import onnx
from transformers.utils import is_accelerate_available, is_torch_available
from ...onnx import remove_duplicate_weights_from_tied_info
from ...onnx import merge_decoders
from ...utils import (
from ...utils import TORCH_MINIMUM_VERSION as GLOBAL_MIN_TORCH_VERSION
from ...utils import TRANSFORMERS_MINIMUM_VERSION as GLOBAL_MIN_TRANSFORMERS_VERSION
from ...utils.doc import add_dynamic_docstring
from ...utils.import_utils import check_if_transformers_greater, is_onnx_available, is_onnxruntime_available
from ..base import ExportConfig
from .constants import ONNX_DECODER_MERGED_NAME, ONNX_DECODER_NAME, ONNX_DECODER_WITH_PAST_NAME
from .model_patcher import ModelPatcher, Seq2SeqModelPatcher
def ordered_inputs(self, model: Union['PreTrainedModel', 'TFPreTrainedModel']) -> Dict[str, Dict[int, str]]:
    """
        Re-orders the inputs using the model forward pass signature.

        Args:
            model ([`transformers.PreTrainedModel`] or [`transformers.TFPreTrainedModel`]):
                The model for which we will use the OnnxConfig.

        Returns:
            `Dict[str, Dict[int, str]]`: The properly ordered inputs.
        """
    inputs = self.inputs
    inputs = self.rename_ambiguous_inputs(inputs)
    ordered_inputs = {}
    if hasattr(model, 'forward'):
        sig = inspect.signature(model.forward)
    else:
        sig = inspect.signature(model.call)
    for param in sig.parameters:
        param_regex = re.compile(f'{param}(\\..*)?$')
        to_insert = []
        for name, dynamic_axes in inputs.items():
            if re.match(param_regex, name):
                to_insert.append((name, dynamic_axes))
        for name, dynamic_axes in to_insert:
            name = self.torch_to_onnx_input_map.get(name, name)
            ordered_inputs[name] = dynamic_axes
    return ordered_inputs