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
def get_decoder_models_for_export(model: Union['PreTrainedModel', 'TFPreTrainedModel'], config: 'OnnxConfig', legacy: bool=False) -> Dict[str, Tuple[Union['PreTrainedModel', 'TFPreTrainedModel'], 'OnnxConfig']]:
    """
    Returns two versions of the decoder that can be used together to perform fast generation:

        1. The first one takes regular inputs, and outputs the result along with past key/values.
        2. The second one takes regular inputs and past key/values, and outputs the result along with the updated past
        key/values.


    Args:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model to export.
        config ([`~exporters.onnx.config.OnnxConfig`]):
            The ONNX configuration associated with the exported model.

    Returns:
        `Dict[str, Tuple[Union[PreTrainedModel, TFPreTrainedModel], OnnxConfig]]: A Dict containing the model and
        onnx configs for the encoder and decoder parts of the model.
    """
    models_for_export = _get_submodels_for_export_decoder(model, use_past=config.use_past, legacy=legacy)
    onnx_kwargs = {'task': config.task, 'float_dtype': config.float_dtype, 'int_dtype': config.int_dtype, 'legacy': legacy}
    if legacy:
        onnx_config = config.__class__(model.config, use_past=config.use_past, use_past_in_inputs=False, **onnx_kwargs)
        models_for_export[ONNX_DECODER_NAME] = (models_for_export[ONNX_DECODER_NAME], onnx_config)
        if config.use_past:
            onnx_config_with_past = config.__class__(model.config, use_past=True, use_past_in_inputs=True, **onnx_kwargs)
            models_for_export[ONNX_DECODER_WITH_PAST_NAME] = (models_for_export[ONNX_DECODER_WITH_PAST_NAME], onnx_config_with_past)
    else:
        onnx_config = config.__class__(model.config, use_past=config.use_past, use_past_in_inputs=config.use_past, **onnx_kwargs)
        models_for_export['model'] = (models_for_export['model'], onnx_config)
    return models_for_export