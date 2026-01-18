from collections import OrderedDict
from typing import Any, Mapping, Optional
from ... import PreTrainedTokenizer
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from ...onnx.utils import compute_effective_axis_dimension
from ...utils import TensorType, is_torch_available, logging
def generate_dummy_inputs(self, tokenizer: PreTrainedTokenizer, batch_size: int=-1, seq_length: int=-1, is_pair: bool=False, framework: Optional[TensorType]=None) -> Mapping[str, Any]:
    if self.task in ['default', 'seq2seq-lm']:
        common_inputs = self._generate_dummy_inputs_for_default_and_seq2seq_lm(tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework)
    else:
        common_inputs = self._generate_dummy_inputs_for_causal_lm(tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework)
    return common_inputs