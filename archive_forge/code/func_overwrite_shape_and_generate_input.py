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
def overwrite_shape_and_generate_input(self, dummy_input_gen: 'DummyInputGenerator', input_name: str, framework: str, input_shapes: Dict):
    """
        The shape passed to the dummy input generator may not always be correct for all of the inputs it manages. This method allows
        to overwrite some shapes, and generate the dummy input. This should probably be refactored more elegantly.
        """
    if self.use_past and self.use_past_in_inputs and (self.use_cache_branch is not False) and (input_name in ['decoder_input_ids', 'input_ids', 'position_ids']) and (self.task == 'text-generation' and self.legacy or self.task != 'text-generation'):
        sequence_length = dummy_input_gen.sequence_length
        dummy_input_gen.sequence_length = 1
        dummy_input = dummy_input_gen.generate(input_name, framework=framework, int_dtype=self.int_dtype, float_dtype=self.float_dtype)
        dummy_input_gen.sequence_length = sequence_length
    else:
        dummy_input = dummy_input_gen.generate(input_name, framework=framework, int_dtype=self.int_dtype, float_dtype=self.float_dtype)
    return dummy_input