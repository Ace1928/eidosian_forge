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
@staticmethod
def tensorrt(per_channel: bool=True, nodes_to_quantize: Optional[List[str]]=None, nodes_to_exclude: Optional[List[str]]=None, operators_to_quantize: Optional[List[str]]=None) -> QuantizationConfig:
    """
        Creates a [`~onnxruntime.QuantizationConfig`] fit for TensorRT static quantization, targetting NVIDIA GPUs.

        Args:
            per_channel (`bool`, defaults to `True`):
                Whether we should quantize per-channel (also known as "per-row"). Enabling this can
                increase overall accuracy while making the quantized model heavier.
            nodes_to_quantize (`Optional[List[str]]`, defaults to `None`):
                Specific nodes to quantize. If `None`, all nodes being operators from `operators_to_quantize` will be quantized.
            nodes_to_exclude (`Optional[List[str]]`, defaults to `None`):
                Specific nodes to exclude from quantization. The list of nodes in a model can be found loading the ONNX model through onnx.load, or through visual inspection with [netron](https://github.com/lutzroeder/netron).
            operators_to_quantize (`Optional[List[str]]`, defaults to `None`):
                Type of nodes to perform quantization on. By default, all the quantizable operators will be quantized. Quantizable operators can be found at https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/registry.py.
        """
    format, mode, operators_to_quantize = default_quantization_parameters(is_static=True, operators_to_quantize=operators_to_quantize)
    return QuantizationConfig(is_static=True, format=format, mode=mode, activations_dtype=QuantType.QInt8, activations_symmetric=True, weights_dtype=QuantType.QInt8, weights_symmetric=True, per_channel=per_channel, reduce_range=False, nodes_to_quantize=nodes_to_quantize or [], nodes_to_exclude=nodes_to_exclude or [], operators_to_quantize=operators_to_quantize, qdq_add_pair_to_weight=True, qdq_dedicated_pair=True)