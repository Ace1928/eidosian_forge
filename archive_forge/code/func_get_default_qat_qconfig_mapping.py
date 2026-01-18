from __future__ import annotations
from collections import OrderedDict
from typing import Any, Callable, Dict, Tuple, Union, List
import torch
from .fake_quantize import (
from .observer import (
from .qconfig import (
def get_default_qat_qconfig_mapping(backend='x86', version=1) -> QConfigMapping:
    """
    Return the default QConfigMapping for quantization aware training.

    Args:
      * ``backend`` (str) : the quantization backend for the default qconfig mapping, should be
         one of ["x86" (default), "fbgemm", "qnnpack", "onednn"]
      * ``version`` (int) : the version for the default qconfig mapping
    """
    return _get_default_qconfig_mapping(True, backend, version)