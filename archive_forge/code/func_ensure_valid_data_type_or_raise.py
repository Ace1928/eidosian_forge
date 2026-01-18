import os
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datasets import Dataset
from packaging.version import Version, parse
from onnxruntime import __version__ as ort_version
from onnxruntime.quantization import CalibraterBase, CalibrationMethod, QuantFormat, QuantizationMode, QuantType
from onnxruntime.quantization.calibrate import create_calibrator
from onnxruntime.quantization.registry import IntegerOpsRegistry, QDQRegistry, QLinearOpsRegistry
from onnxruntime.transformers.fusion_options import FusionOptions
from ..configuration_utils import BaseConfig
from ..utils import logging
def ensure_valid_data_type_or_raise(use_static_quantization: bool, activations_dtype: QuantType, weights_dtype: QuantType):
    if not use_static_quantization and activations_dtype == QuantType.QInt8:
        raise ValueError('Invalid combination of use_static_quantization = False and activations_dtype = QuantType.QInt8. OnnxRuntime dynamic quantization requires activations_dtype = QuantType.QUInt8')
    if use_static_quantization and activations_dtype == QuantType.QInt8 and (weights_dtype == QuantType.QUInt8):
        raise ValueError('Invalid combination of use_static_quantization = True, activations_dtype = QuantType.QInt8 and weights_dtype = QuantType.QUInt8.OnnxRuntime static quantization does not support activations_dtype = QuantType.QInt8 with weights_dtype = QuantType.QUInt8.')