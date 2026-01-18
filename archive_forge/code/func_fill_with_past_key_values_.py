from collections import OrderedDict
from typing import Any, Mapping, Optional
from ... import PreTrainedTokenizer
from ...configuration_utils import PretrainedConfig
from ...file_utils import TensorType, is_torch_available
from ...onnx import OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from ...onnx.utils import compute_effective_axis_dimension
from ...utils import logging
def fill_with_past_key_values_(self, inputs_or_outputs: Mapping[str, Mapping[int, str]], direction: str):
    if direction not in ['inputs', 'outputs']:
        raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')
    name = 'past_key_values' if direction == 'inputs' else 'present'
    _, num_decoder_layers = self.num_layers
    encoder_sequence = 'past_encoder_sequence'
    decoder_sequence = 'past_decoder_sequence' if direction == 'inputs' else 'past_decoder_sequence + sequence'
    for i in range(num_decoder_layers):
        inputs_or_outputs[f'{name}.{i}.decoder.key'] = {0: 'batch', 2: decoder_sequence}
        inputs_or_outputs[f'{name}.{i}.decoder.value'] = {0: 'batch', 2: decoder_sequence}
        inputs_or_outputs[f'{name}.{i}.encoder.key'] = {0: 'batch', 2: encoder_sequence}
        inputs_or_outputs[f'{name}.{i}.encoder.value'] = {0: 'batch', 2: encoder_sequence}