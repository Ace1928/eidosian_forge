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
def avx2(is_static: bool, use_symmetric_activations: bool=False, use_symmetric_weights: bool=True, per_channel: bool=True, reduce_range: bool=False, nodes_to_quantize: Optional[List[str]]=None, nodes_to_exclude: Optional[List[str]]=None, operators_to_quantize: Optional[List[str]]=None) -> QuantizationConfig:
    """
        Creates a [`~onnxruntime.QuantizationConfig`] fit for CPU with AVX2 instruction set.

        Args:
            is_static (`bool`):
                Boolean flag to indicate whether we target static or dynamic quantization.
            use_symmetric_activations (`bool`, defaults to `False`):
                Whether to use symmetric quantization for activations.
            use_symmetric_weights (`bool`, defaults to `True`):
                Whether to use symmetric quantization for weights.
            per_channel (`bool`, defaults to `True`):
                Whether we should quantize per-channel (also known as "per-row"). Enabling this can
                increase overall accuracy while making the quantized model heavier.
            reduce_range (`bool`, defaults to `False`):
                Indicate whether to use 8-bits integers (False) or reduce-range 7-bits integers (True).
                As a baseline, it is always recommended testing with full range (reduce_range = False) and then, if
                accuracy drop is significant, to try with reduced range (reduce_range = True).
                Intel's CPUs using AVX512 (non VNNI) can suffer from saturation issue when invoking
                the VPMADDUBSW instruction. To counter this, one should use 7-bits rather than 8-bits integers.
            nodes_to_quantize (`Optional[List[str]]`, defaults to `None`):
                Specific nodes to quantize. If `None`, all nodes being operators from `operators_to_quantize` will be quantized.
            nodes_to_exclude (`Optional[List[str]]`, defaults to `None`):
                Specific nodes to exclude from quantization. The list of nodes in a model can be found loading the ONNX model through onnx.load, or through visual inspection with [netron](https://github.com/lutzroeder/netron).
            operators_to_quantize (`Optional[List[str]]`, defaults to `None`):
                Type of nodes to perform quantization on. By default, all the quantizable operators will be quantized. Quantizable operators can be found at https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/registry.py.
        """
    format, mode, operators_to_quantize = default_quantization_parameters(is_static, operators_to_quantize=operators_to_quantize)
    return QuantizationConfig(is_static=is_static, format=format, mode=mode, activations_dtype=QuantType.QUInt8, activations_symmetric=use_symmetric_activations, weights_dtype=QuantType.QUInt8, weights_symmetric=use_symmetric_weights, per_channel=per_channel, reduce_range=reduce_range, nodes_to_quantize=nodes_to_quantize or [], nodes_to_exclude=nodes_to_exclude or [], operators_to_quantize=operators_to_quantize)