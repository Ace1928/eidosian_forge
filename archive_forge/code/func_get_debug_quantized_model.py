import collections
import csv
import re
from typing import (Any, Callable, Dict, IO, Iterable, List, Mapping, Optional,
import numpy as np
from tensorflow.lite.python import convert
from tensorflow.lite.python import interpreter as _interpreter
from tensorflow.lite.python.metrics import metrics as metrics_stub  # type: ignore
from tensorflow.python.util import tf_export
def get_debug_quantized_model(self) -> bytes:
    """Returns an instrumented quantized model.

    Convert the quantized model with the initialized converter and
    return bytes for model. The model will be instrumented with numeric
    verification operations and should only be used for debugging.

    Returns:
      Model bytes corresponding to the model.
    Raises:
      ValueError: if converter is not passed to the debugger.
    """
    return self._get_quantized_model(is_debug=True)