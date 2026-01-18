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
def get_stable_diffusion_models_for_export(pipeline: 'StableDiffusionPipeline', int_dtype: str='int64', float_dtype: str='fp32') -> Dict[str, Tuple[Union['PreTrainedModel', 'ModelMixin'], 'OnnxConfig']]:
    """
    Returns the components of a Stable Diffusion model and their subsequent onnx configs.

    Args:
        pipeline ([`StableDiffusionPipeline`]):
            The model to export.
        int_dtype (`str`, defaults to `"int64"`):
            The data type of integer tensors, could be ["int64", "int32", "int8"], default to "int64".
        float_dtype (`str`, defaults to `"fp32"`):
            The data type of float tensors, could be ["fp32", "fp16", "bf16"], default to "fp32".

    Returns:
        `Dict[str, Tuple[Union[`PreTrainedModel`, `TFPreTrainedModel`], `OnnxConfig`]: A Dict containing the model and
        onnx configs for the different components of the model.
    """
    models_for_export = _get_submodels_for_export_stable_diffusion(pipeline)
    if 'text_encoder' in models_for_export:
        text_encoder_config_constructor = TasksManager.get_exporter_config_constructor(model=pipeline.text_encoder, exporter='onnx', library_name='diffusers', task='feature-extraction')
        text_encoder_onnx_config = text_encoder_config_constructor(pipeline.text_encoder.config, int_dtype=int_dtype, float_dtype=float_dtype)
        models_for_export['text_encoder'] = (models_for_export['text_encoder'], text_encoder_onnx_config)
    onnx_config_constructor = TasksManager.get_exporter_config_constructor(model=pipeline.unet, exporter='onnx', library_name='diffusers', task='semantic-segmentation', model_type='unet')
    unet_onnx_config = onnx_config_constructor(pipeline.unet.config, int_dtype=int_dtype, float_dtype=float_dtype)
    models_for_export['unet'] = (models_for_export['unet'], unet_onnx_config)
    vae_encoder = models_for_export['vae_encoder']
    vae_config_constructor = TasksManager.get_exporter_config_constructor(model=vae_encoder, exporter='onnx', library_name='diffusers', task='semantic-segmentation', model_type='vae-encoder')
    vae_onnx_config = vae_config_constructor(vae_encoder.config, int_dtype=int_dtype, float_dtype=float_dtype)
    models_for_export['vae_encoder'] = (vae_encoder, vae_onnx_config)
    vae_decoder = models_for_export['vae_decoder']
    vae_config_constructor = TasksManager.get_exporter_config_constructor(model=vae_decoder, exporter='onnx', library_name='diffusers', task='semantic-segmentation', model_type='vae-decoder')
    vae_onnx_config = vae_config_constructor(vae_decoder.config, int_dtype=int_dtype, float_dtype=float_dtype)
    models_for_export['vae_decoder'] = (vae_decoder, vae_onnx_config)
    if 'text_encoder_2' in models_for_export:
        onnx_config_constructor = TasksManager.get_exporter_config_constructor(model=pipeline.text_encoder_2, exporter='onnx', library_name='diffusers', task='feature-extraction', model_type='clip-text-with-projection')
        onnx_config = onnx_config_constructor(pipeline.text_encoder_2.config, int_dtype=int_dtype, float_dtype=float_dtype)
        models_for_export['text_encoder_2'] = (models_for_export['text_encoder_2'], onnx_config)
    return models_for_export