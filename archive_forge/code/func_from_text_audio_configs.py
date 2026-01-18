import os
from typing import Union
from ...configuration_utils import PretrainedConfig
from ...utils import logging
@classmethod
def from_text_audio_configs(cls, text_config: ClapTextConfig, audio_config: ClapAudioConfig, **kwargs):
    """
        Instantiate a [`ClapConfig`] (or a derived class) from clap text model configuration and clap audio model
        configuration.

        Returns:
            [`ClapConfig`]: An instance of a configuration object
        """
    return cls(text_config=text_config.to_dict(), audio_config=audio_config.to_dict(), **kwargs)