from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto.configuration_auto import AutoConfig
@property
def sampling_rate(self):
    return self.audio_encoder.sampling_rate