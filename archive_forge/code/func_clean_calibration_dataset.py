import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union
import onnx
from datasets import Dataset, load_dataset
from packaging.version import Version, parse
from transformers import AutoConfig
from onnxruntime import __version__ as ort_version
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantizationMode, QuantType
from onnxruntime.quantization.onnx_quantizer import ONNXQuantizer
from onnxruntime.quantization.qdq_quantizer import QDQQuantizer
from ..quantization_base import OptimumQuantizer
from ..utils.save_utils import maybe_save_preprocessors
from . import ORTQuantizableOperator
from .configuration import CalibrationConfig, ORTConfig, QuantizationConfig
from .modeling_ort import ORTModel
from .modeling_seq2seq import ORTModelForConditionalGeneration
from .preprocessors import QuantizationPreprocessor
def clean_calibration_dataset(self, dataset: Dataset) -> Dataset:
    model = onnx.load(self.onnx_model_path)
    model_inputs = {input.name for input in model.graph.input}
    ignored_columns = list(set(dataset.column_names) - model_inputs)
    return dataset.remove_columns(ignored_columns)