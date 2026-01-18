import copy
import enum
import gc
import inspect
import itertools
import os
import re
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import onnx
from transformers.utils import is_accelerate_available, is_torch_available
from ...onnx import remove_duplicate_weights_from_tied_info
from ...onnx import merge_decoders
from ...utils import (
from ...utils import TORCH_MINIMUM_VERSION as GLOBAL_MIN_TORCH_VERSION
from ...utils import TRANSFORMERS_MINIMUM_VERSION as GLOBAL_MIN_TRANSFORMERS_VERSION
from ...utils.doc import add_dynamic_docstring
from ...utils.import_utils import check_if_transformers_greater, is_onnx_available, is_onnxruntime_available
from ..base import ExportConfig
from .constants import ONNX_DECODER_MERGED_NAME, ONNX_DECODER_NAME, ONNX_DECODER_WITH_PAST_NAME
from .model_patcher import ModelPatcher, Seq2SeqModelPatcher
def flatten_seq2seq_past_key_values(self, flattened_output, name, idx, t):
    if len(t) not in [2, 4]:
        raise ValueError('past_key_values to flatten should be of length 2 (self-attention only) or 4 (self and cross attention).')
    if len(t) == 2:
        flattened_output[f'{name}.{idx}.decoder.key'] = t[0]
        flattened_output[f'{name}.{idx}.decoder.value'] = t[1]
    if len(t) == 4:
        flattened_output[f'{name}.{idx}.encoder.key'] = t[2]
        flattened_output[f'{name}.{idx}.encoder.value'] = t[3]