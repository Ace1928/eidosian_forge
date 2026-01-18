import logging
import re
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import torch
from huggingface_hub import HfApi, HfFolder, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from transformers import (
from transformers.file_utils import add_end_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.modeling_outputs import (
import onnxruntime as ort
from ..exporters import TasksManager
from ..exporters.onnx import main_export
from ..modeling_base import FROM_PRETRAINED_START_DOCSTRING, OptimizedModel
from ..onnx.utils import _get_external_data_paths
from ..utils.file_utils import find_files_matching_pattern
from ..utils.save_utils import maybe_load_preprocessors, maybe_save_preprocessors
from .io_binding import IOBindingHelper, TypeHelper
from .utils import (
@staticmethod
def infer_onnx_filename(model_name_or_path: Union[str, Path], patterns: List[str], argument_name: str, subfolder: str='', use_auth_token: Optional[Union[bool, str]]=None, revision: Optional[str]=None, fail_if_not_found: bool=True) -> str:
    onnx_files = []
    for pattern in patterns:
        onnx_files = find_files_matching_pattern(model_name_or_path, pattern, glob_pattern='**/*.onnx', subfolder=subfolder, use_auth_token=use_auth_token, revision=revision)
        if onnx_files:
            break
    path = model_name_or_path
    if subfolder != '':
        path = f'{path}/{subfolder}'
    if len(onnx_files) == 0:
        if fail_if_not_found:
            raise FileNotFoundError(f'Could not find any ONNX model file for the regex {patterns} in {path}.')
        return None
    elif len(onnx_files) > 1:
        if argument_name is not None:
            raise RuntimeError(f'Too many ONNX model files were found in {path}, specify which one to load by using the {argument_name} argument.')
    return onnx_files[0]