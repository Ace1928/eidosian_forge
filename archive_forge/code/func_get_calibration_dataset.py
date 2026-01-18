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
def get_calibration_dataset(self, dataset_name: str, num_samples: int=100, dataset_config_name: Optional[str]=None, dataset_split: Optional[str]=None, preprocess_function: Optional[Callable]=None, preprocess_batch: bool=True, seed: int=2016, use_auth_token: bool=False) -> Dataset:
    """
        Creates the calibration `datasets.Dataset` to use for the post-training static quantization calibration step.

        Args:
            dataset_name (`str`):
                The dataset repository name on the Hugging Face Hub or path to a local directory containing data files
                to load to use for the calibration step.
            num_samples (`int`, defaults to 100):
                The maximum number of samples composing the calibration dataset.
            dataset_config_name (`Optional[str]`, defaults to `None`):
                The name of the dataset configuration.
            dataset_split (`Optional[str]`, defaults to `None`):
                Which split of the dataset to use to perform the calibration step.
            preprocess_function (`Optional[Callable]`, defaults to `None`):
                Processing function to apply to each example after loading dataset.
            preprocess_batch (`bool`, defaults to `True`):
                Whether the `preprocess_function` should be batched.
            seed (`int`, defaults to 2016):
                The random seed to use when shuffling the calibration dataset.
            use_auth_token (`bool`, defaults to `False`):
                Whether to use the token generated when running `transformers-cli login` (necessary for some datasets
                like ImageNet).
        Returns:
            The calibration `datasets.Dataset` to use for the post-training static quantization calibration
            step.
        """
    if dataset_name is None:
        raise ValueError('ORTQuantizer: Static quantization calibration step requires a dataset_name if no calib_dataset is provided.')
    calib_dataset = load_dataset(dataset_name, name=dataset_config_name, split=dataset_split, use_auth_token=use_auth_token)
    if num_samples is not None:
        num_samples = min(num_samples, len(calib_dataset))
        calib_dataset = calib_dataset.shuffle(seed=seed).select(range(num_samples))
    if preprocess_function is not None:
        processed_calib_dataset = calib_dataset.map(preprocess_function, batched=preprocess_batch)
    else:
        processed_calib_dataset = calib_dataset
    return self.clean_calibration_dataset(processed_calib_dataset)