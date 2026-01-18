import copy
import json
import os
import warnings
from typing import Any, Dict, Optional, Union
from .. import __version__
from ..configuration_utils import PretrainedConfig
from ..utils import (
@classmethod
def from_model_config(cls, model_config: PretrainedConfig) -> 'GenerationConfig':
    """
        Instantiates a [`GenerationConfig`] from a [`PretrainedConfig`]. This function is useful to convert legacy
        [`PretrainedConfig`] objects, which may contain generation parameters, into a stand-alone [`GenerationConfig`].

        Args:
            model_config (`PretrainedConfig`):
                The model config that will be used to instantiate the generation config.

        Returns:
            [`GenerationConfig`]: The configuration object instantiated from those parameters.
        """
    config_dict = model_config.to_dict()
    config_dict.pop('_from_model_config', None)
    config = cls.from_dict(config_dict, return_unused_kwargs=False, _from_model_config=True)
    for decoder_name in ('decoder', 'generator', 'text_config'):
        if decoder_name in config_dict:
            default_generation_config = GenerationConfig()
            decoder_config = config_dict[decoder_name]
            for attr in config.to_dict().keys():
                if attr in decoder_config and getattr(config, attr) == getattr(default_generation_config, attr):
                    setattr(config, attr, decoder_config[attr])
    config._original_object_hash = hash(config)
    return config