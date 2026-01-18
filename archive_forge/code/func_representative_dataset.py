from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple
import numpy as np
from transformers import PreTrainedTokenizerBase
from transformers.utils import is_tf_available
from ...utils import logging
from ...utils.preprocessing import Preprocessor, TaskProcessorsManager
from ..error_utils import AtolError, OutputMatchError, ShapeError
from .base import QuantizationApproach, QuantizationApproachNotSupported
def representative_dataset():
    for sig_name, tf_function in signatures.items():
        inputs_to_keep = None
        for example in dataset:
            if inputs_to_keep is None:
                args, kwargs = tf_function.structured_input_signature
                args_to_keep = {input_.name for input_ in args if input_.name in example}
                kwargs_to_keep = {input_.name for input_ in kwargs.values() if input_.name in example}
                inputs_to_keep = args_to_keep | kwargs_to_keep
            yield (sig_name, {name: value for name, value in example.items() if name in inputs_to_keep})