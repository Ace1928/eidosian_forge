from collections import OrderedDict
from typing import Any, Mapping, Optional
from ... import PreTrainedTokenizer
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from ...onnx.utils import compute_effective_axis_dimension
from ...utils import TensorType, is_torch_available, logging
@property
def outputs(self) -> Mapping[str, Mapping[int, str]]:
    if self.task in ['default', 'seq2seq-lm']:
        common_outputs = super().outputs
    else:
        common_outputs = super(OnnxConfigWithPast, self).outputs
        if self.use_past:
            num_encoder_layers, _ = self.num_layers
            for i in range(num_encoder_layers):
                common_outputs[f'present.{i}.key'] = {0: 'batch', 2: 'past_sequence + sequence'}
                common_outputs[f'present.{i}.value'] = {0: 'batch', 2: 'past_sequence + sequence'}
    return common_outputs