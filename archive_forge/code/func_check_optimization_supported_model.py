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
def check_optimization_supported_model(cls, model_type: str, optimization_config):
    supported_model_types_for_optimization = ['bart', 'bert', 'gpt2', 'tnlr', 't5', 'unet', 'vae', 'clip', 'vit', 'swin']
    model_type = model_type.replace('_', '-')
    if model_type not in cls._conf or cls._conf[model_type] not in supported_model_types_for_optimization:
        raise NotImplementedError(f"ONNX Runtime doesn't support the graph optimization of {model_type} yet. Only {list(cls._conf.keys())} are supported. If you want to support {model_type} please propose a PR or open up an issue in ONNX Runtime: https://github.com/microsoft/onnxruntime.")