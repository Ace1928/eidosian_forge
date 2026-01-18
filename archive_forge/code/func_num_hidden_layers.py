from typing import Callable, Optional, Union
from ...configuration_utils import PretrainedConfig
from ...utils import logging
@num_hidden_layers.setter
def num_hidden_layers(self, value):
    raise NotImplementedError('This model does not support the setting of `num_hidden_layers`. Please set `num_encoder_layers` and `num_decoder_layers`.')