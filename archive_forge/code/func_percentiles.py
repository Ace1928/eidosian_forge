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
def percentiles(dataset: Dataset, num_bins: int=2048, percentile: float=99.999) -> CalibrationConfig:
    """
        Args:
            dataset (`Dataset`):
                The dataset to use when performing the calibration step.
            num_bins (`int`):
                The number of bins to use when creating the histogram.
            percentile (`float`):
                The percentile to use when computing the activations quantization ranges.

        Returns:
            The calibration configuration.
        """
    if parse(ort_version) < Version('1.11.0'):
        raise NotImplementedError('Percentile calibration method is only implemented for onnxruntime >= 1.11.0')
    if num_bins <= 0:
        raise ValueError(f'Invalid value num_bins ({num_bins}) should be >= 1')
    if not 0 <= percentile <= 100:
        raise ValueError(f'Invalid value percentile ({percentile}) should be within  [0, 100]')
    return CalibrationConfig(dataset_name=dataset.info.builder_name, dataset_config_name=dataset.info.config_name, dataset_split=str(dataset.split), dataset_num_samples=dataset.num_rows, method=CalibrationMethod.Percentile, num_bins=num_bins, percentile=percentile)