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
@classmethod
def get_model_ort_type(cls, model_type: str) -> str:
    model_type = model_type.replace('_', '-')
    cls.check_supported_model(model_type)
    return cls._conf[model_type]