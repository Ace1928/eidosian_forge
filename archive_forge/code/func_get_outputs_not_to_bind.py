from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, Optional, Set, Tuple, Union
import numpy as np
import torch
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from onnxruntime import InferenceSession
from ..utils import NormalizedConfigManager
from ..utils.logging import warn_once
from .utils import get_ordered_input_names, logging
def get_outputs_not_to_bind(self, use_merged_cache: bool) -> Set[str]:
    result = {output_name for output_name in self.output_names if not output_name.startswith('present') and output_name not in {'loss', 'logits'}}
    if use_merged_cache is True:
        result = result.union(self.past_key_values_cross_attention_output_names)
    return result