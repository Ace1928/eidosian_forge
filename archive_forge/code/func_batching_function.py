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
def batching_function(examples):
    return {column_name: [examples[column_name]] for column_name in examples.keys()}