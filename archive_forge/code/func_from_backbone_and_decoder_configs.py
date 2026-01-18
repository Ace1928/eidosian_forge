from typing import Dict, Optional
from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING
from ..detr import DetrConfig
from ..swin import SwinConfig
@classmethod
def from_backbone_and_decoder_configs(cls, backbone_config: PretrainedConfig, decoder_config: PretrainedConfig, **kwargs):
    """Instantiate a [`MaskFormerConfig`] (or a derived class) from a pre-trained backbone model configuration and DETR model
        configuration.

            Args:
                backbone_config ([`PretrainedConfig`]):
                    The backbone configuration.
                decoder_config ([`PretrainedConfig`]):
                    The transformer decoder configuration to use.

            Returns:
                [`MaskFormerConfig`]: An instance of a configuration object
        """
    return cls(backbone_config=backbone_config, decoder_config=decoder_config, **kwargs)