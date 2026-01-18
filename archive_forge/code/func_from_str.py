import collections
import csv
import importlib
import json
import os
import pickle
import sys
import traceback
import types
import warnings
from abc import ABC, abstractmethod
from collections import UserDict
from contextlib import contextmanager
from os.path import abspath, exists
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from ..dynamic_module_utils import custom_object_save
from ..feature_extraction_utils import PreTrainedFeatureExtractor
from ..image_processing_utils import BaseImageProcessor
from ..modelcard import ModelCard
from ..models.auto.configuration_auto import AutoConfig
from ..tokenization_utils import PreTrainedTokenizer
from ..utils import (
@staticmethod
def from_str(format: str, output_path: Optional[str], input_path: Optional[str], column: Optional[str], overwrite=False) -> 'PipelineDataFormat':
    """
        Creates an instance of the right subclass of [`~pipelines.PipelineDataFormat`] depending on `format`.

        Args:
            format (`str`):
                The format of the desired pipeline. Acceptable values are `"json"`, `"csv"` or `"pipe"`.
            output_path (`str`, *optional*):
                Where to save the outgoing data.
            input_path (`str`, *optional*):
                Where to look for the input data.
            column (`str`, *optional*):
                The column to read.
            overwrite (`bool`, *optional*, defaults to `False`):
                Whether or not to overwrite the `output_path`.

        Returns:
            [`~pipelines.PipelineDataFormat`]: The proper data format.
        """
    if format == 'json':
        return JsonPipelineDataFormat(output_path, input_path, column, overwrite=overwrite)
    elif format == 'csv':
        return CsvPipelineDataFormat(output_path, input_path, column, overwrite=overwrite)
    elif format == 'pipe':
        return PipedPipelineDataFormat(output_path, input_path, column, overwrite=overwrite)
    else:
        raise KeyError(f'Unknown reader {format} (Available reader are json/csv/pipe)')