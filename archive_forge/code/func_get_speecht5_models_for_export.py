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
def get_speecht5_models_for_export(model: Union['PreTrainedModel', 'TFPreTrainedModel'], config: 'OnnxConfig', model_kwargs: Optional[Dict]):
    if model_kwargs is None or 'vocoder' not in model_kwargs:
        raise ValueError('The ONNX export of SpeechT5 requires a vocoder. Please pass `--model-kwargs \'{"vocoder": "vocoder_model_name_or_path"}\'` from the command line, or `model_kwargs={"vocoder": "vocoder_model_name_or_path"}` if calling main_export.')
    models_for_export = {}
    models_for_export['encoder_model'] = model
    models_for_export['decoder_model'] = model
    if config.variant == 'with-past':
        models_for_export['decoder_with_past_model'] = model
    vocoder = SpeechT5HifiGan.from_pretrained(model_kwargs['vocoder']).eval()
    model_kwargs['vocoder_model'] = vocoder
    models_for_export['decoder_postnet_and_vocoder'] = model
    encoder_onnx_config = config.with_behavior('encoder')
    use_past = config.variant == 'with-past'
    decoder_onnx_config = config.with_behavior('decoder', use_past=use_past, use_past_in_inputs=False)
    models_for_export[ONNX_ENCODER_NAME] = (models_for_export[ONNX_ENCODER_NAME], encoder_onnx_config)
    models_for_export[ONNX_DECODER_NAME] = (models_for_export[ONNX_DECODER_NAME], decoder_onnx_config)
    if config.variant == 'with-past':
        decoder_onnx_config_with_past = config.with_behavior('decoder', use_past=True, use_past_in_inputs=True)
        models_for_export[ONNX_DECODER_WITH_PAST_NAME] = (models_for_export[ONNX_DECODER_WITH_PAST_NAME], decoder_onnx_config_with_past)
    postnet_and_vocoder_onnx_config = config.__class__(config._config, task=config.task, int_dtype=config.int_dtype, float_dtype=config.float_dtype, use_past=use_past, use_past_in_inputs=False, behavior=config._behavior, preprocessors=config._preprocessors, is_postnet_and_vocoder=True, legacy=config.legacy)
    postnet_and_vocoder_onnx_config.variant = config.variant
    models_for_export['decoder_postnet_and_vocoder'] = (models_for_export['decoder_postnet_and_vocoder'], postnet_and_vocoder_onnx_config)
    return models_for_export