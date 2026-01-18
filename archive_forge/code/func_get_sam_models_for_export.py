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
def get_sam_models_for_export(model: Union['PreTrainedModel', 'TFPreTrainedModel'], config: 'OnnxConfig'):
    models_for_export = _get_submodels_for_export_sam(model, config.variant)
    if config.variant == 'monolith':
        onnx_config = config.__class__(model.config, task=config.task, legacy=config.legacy)
        models_for_export['model'] = (models_for_export['model'], onnx_config)
    else:
        vision_encoder_onnx_config = config.__class__(model.config, task=config.task, variant=config.variant, vision_encoder=True, legacy=config.legacy)
        prompt_encoder_mask_decoder_onnx_config = config.__class__(model.config, task=config.task, variant=config.variant, vision_encoder=False, legacy=config.legacy)
        models_for_export['vision_encoder'] = (models_for_export['vision_encoder'], vision_encoder_onnx_config)
        models_for_export['prompt_encoder_mask_decoder'] = (models_for_export['prompt_encoder_mask_decoder'], prompt_encoder_mask_decoder_onnx_config)
    return models_for_export