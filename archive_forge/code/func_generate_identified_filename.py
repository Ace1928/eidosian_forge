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
def generate_identified_filename(filename, identifier):
    return filename.parent.joinpath(filename.stem + identifier).with_suffix(filename.suffix)