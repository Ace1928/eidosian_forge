import copy
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from packaging import version
from transformers.models.speecht5.modeling_speecht5 import SpeechT5HifiGan
from transformers.utils import is_tf_available, is_torch_available
from ...utils import (
from ...utils.import_utils import _diffusers_version
from ..tasks import TasksManager
from .constants import ONNX_DECODER_NAME, ONNX_DECODER_WITH_PAST_NAME, ONNX_ENCODER_NAME
def get_encoder_decoder_models_for_export(model: Union['PreTrainedModel', 'TFPreTrainedModel'], config: 'OnnxConfig') -> Dict[str, Tuple[Union['PreTrainedModel', 'TFPreTrainedModel'], 'OnnxConfig']]:
    """
    Returns the encoder and decoder parts of the model and their subsequent onnx configs.

    Args:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model to export.
        config ([`~exporters.onnx.config.OnnxConfig`]):
            The ONNX configuration associated with the exported model.

    Returns:
        `Dict[str, Tuple[Union[`PreTrainedModel`, `TFPreTrainedModel`], `OnnxConfig`]: A Dict containing the model and
        onnx configs for the encoder and decoder parts of the model.
    """
    models_for_export = _get_submodels_for_export_encoder_decoder(model, use_past=config.use_past)
    encoder_onnx_config = config.with_behavior('encoder')
    models_for_export[ONNX_ENCODER_NAME] = (models_for_export[ONNX_ENCODER_NAME], encoder_onnx_config)
    decoder_onnx_config = config.with_behavior('decoder', use_past=config.use_past, use_past_in_inputs=False)
    models_for_export[ONNX_DECODER_NAME] = (models_for_export[ONNX_DECODER_NAME], decoder_onnx_config)
    if config.use_past:
        decoder_onnx_config_with_past = config.with_behavior('decoder', use_past=True, use_past_in_inputs=True)
        models_for_export[ONNX_DECODER_WITH_PAST_NAME] = (models_for_export[ONNX_DECODER_WITH_PAST_NAME], decoder_onnx_config_with_past)
    return models_for_export