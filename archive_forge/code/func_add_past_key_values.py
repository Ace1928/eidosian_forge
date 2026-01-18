from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union
from transformers.utils import is_tf_available
from ...onnx import merge_decoders
from ...utils import (
from .base import ConfigBehavior, OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from .constants import ONNX_DECODER_MERGED_NAME, ONNX_DECODER_NAME, ONNX_DECODER_WITH_PAST_NAME
from .model_patcher import DecoderModelPatcher
def add_past_key_values(self, inputs_or_outputs: Dict[str, Dict[int, str]], direction: str):
    if self.is_decoder_with_past:
        return self._decoder_onnx_config.add_past_key_values(inputs_or_outputs, direction)