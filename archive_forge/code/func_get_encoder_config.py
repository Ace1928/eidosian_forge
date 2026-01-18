from typing import TYPE_CHECKING, Any, Mapping, Optional, OrderedDict
from packaging import version
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging
from ..auto.configuration_auto import AutoConfig
def get_encoder_config(self, encoder_config: PretrainedConfig) -> OnnxConfig:
    """
        Returns ONNX encoder config for `VisionEncoderDecoder` model.

        Args:
            encoder_config (`PretrainedConfig`):
                The encoder model's configuration to use when exporting to ONNX.

        Returns:
            [`VisionEncoderDecoderEncoderOnnxConfig`]: An instance of the ONNX configuration object
        """
    return VisionEncoderDecoderEncoderOnnxConfig(encoder_config)