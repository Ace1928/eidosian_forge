import copy
import gc
import multiprocessing as mp
import os
import traceback
from inspect import signature
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import onnx
from transformers.modeling_utils import get_parameter_dtype
from transformers.utils import is_tf_available, is_torch_available
from ...onnx.utils import _get_onnx_external_data_tensors, check_model_uses_external_data
from ...utils import (
from ...utils.modeling_utils import MODEL_TO_PATCH_FOR_PAST
from ...utils.save_utils import maybe_save_preprocessors
from ..error_utils import AtolError, MinimumVersionError, OutputMatchError, ShapeError
from ..tasks import TasksManager
from .base import OnnxConfig
from .constants import UNPICKABLE_ARCHS
from .model_configs import SpeechT5OnnxConfig
from .utils import (
def check_dummy_inputs_are_allowed(model: Union['PreTrainedModel', 'TFPreTrainedModel', 'ModelMixin'], dummy_input_names: Iterable[str]):
    """
    Checks that the dummy inputs from the ONNX config is a subset of the allowed inputs for `model`.
    Args:
        model (`Union[transformers.PreTrainedModel, transformers.TFPreTrainedModel`]):
            The model instance.
        model_inputs (`Iterable[str]`):
            The model input names.
    """
    forward = model.forward if is_torch_available() and isinstance(model, nn.Module) else model.call
    forward_parameters = signature(forward).parameters
    forward_inputs_set = set(forward_parameters.keys())
    dummy_input_names = set(dummy_input_names)
    if not dummy_input_names.issubset(forward_inputs_set):
        raise ValueError(f'Config dummy inputs are not a subset of the model inputs: {dummy_input_names} vs {forward_inputs_set}')