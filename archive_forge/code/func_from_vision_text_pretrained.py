from typing import Optional, Tuple, Union
import torch
from torch import nn
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from ..auto.configuration_auto import AutoConfig
from ..auto.modeling_auto import AutoModel
from ..clip.modeling_clip import CLIPOutput, CLIPVisionConfig, CLIPVisionModel
from .configuration_vision_text_dual_encoder import VisionTextDualEncoderConfig
@classmethod
def from_vision_text_pretrained(cls, vision_model_name_or_path: str=None, text_model_name_or_path: str=None, *model_args, **kwargs) -> PreTrainedModel:
    """
        Params:
            vision_model_name_or_path (`str`, *optional*, defaults to `None`):
                Information necessary to initiate the vision model. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *PyTorch checkpoint folder* (e.g, `./pt_model`). In this case, `from_pt`
                      should be set to `True` and a configuration object should be provided as `config` argument. This
                      loading path is slower than converting the PyTorch checkpoint in a Flax model using the provided
                      conversion scripts and loading the Flax model afterwards.

            text_model_name_or_path (`str`, *optional*):
                Information necessary to initiate the text model. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *PyTorch checkpoint folder* (e.g, `./pt_model`). In this case, `from_pt`
                      should be set to `True` and a configuration object should be provided as `config` argument. This
                      loading path is slower than converting the PyTorch checkpoint in a Flax model using the provided
                      conversion scripts and loading the Flax model afterwards.

            model_args (remaining positional arguments, *optional*):
                All remaning positional arguments will be passed to the underlying model's `__init__` method.

            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`).

                - To update the text configuration, use the prefix *text_* for each configuration parameter.
                - To update the vision configuration, use the prefix *vision_* for each configuration parameter.
                - To update the parent model configuration, do not use a prefix for each configuration parameter.

                Behaves differently depending on whether a `config` is provided or automatically loaded.

        Example:

        ```python
        >>> from transformers import VisionTextDualEncoderModel

        >>> # initialize a model from pretrained ViT and BERT models. Note that the projection layers will be randomly initialized.
        >>> model = VisionTextDualEncoderModel.from_vision_text_pretrained(
        ...     "google/vit-base-patch16-224", "google-bert/bert-base-uncased"
        ... )
        >>> # saving model after fine-tuning
        >>> model.save_pretrained("./vit-bert")
        >>> # load fine-tuned model
        >>> model = VisionTextDualEncoderModel.from_pretrained("./vit-bert")
        ```"""
    kwargs_vision = {argument[len('vision_'):]: value for argument, value in kwargs.items() if argument.startswith('vision_')}
    kwargs_text = {argument[len('text_'):]: value for argument, value in kwargs.items() if argument.startswith('text_')}
    for key in kwargs_vision.keys():
        del kwargs['vision_' + key]
    for key in kwargs_text.keys():
        del kwargs['text_' + key]
    vision_model = kwargs_vision.pop('model', None)
    if vision_model is None:
        if vision_model_name_or_path is None:
            raise ValueError('If `vision_model` is not defined as an argument, a `vision_model_name_or_path` has to be defined')
        if 'config' not in kwargs_vision:
            vision_config = AutoConfig.from_pretrained(vision_model_name_or_path)
        if vision_config.model_type == 'clip':
            kwargs_vision['config'] = vision_config.vision_config
            vision_model = CLIPVisionModel.from_pretrained(vision_model_name_or_path, *model_args, **kwargs_vision)
        else:
            kwargs_vision['config'] = vision_config
            vision_model = AutoModel.from_pretrained(vision_model_name_or_path, *model_args, **kwargs_vision)
    text_model = kwargs_text.pop('model', None)
    if text_model is None:
        if text_model_name_or_path is None:
            raise ValueError('If `text_model` is not defined as an argument, a `text_model_name_or_path` has to be defined')
        if 'config' not in kwargs_text:
            text_config = AutoConfig.from_pretrained(text_model_name_or_path)
            kwargs_text['config'] = text_config
        text_model = AutoModel.from_pretrained(text_model_name_or_path, *model_args, **kwargs_text)
    config = VisionTextDualEncoderConfig.from_vision_text_configs(vision_model.config, text_model.config, **kwargs)
    model = cls(config=config, vision_model=vision_model, text_model=text_model)
    logger.warning("The projection layer and logit scale weights `['visual_projection.weight', 'text_projection.weight', 'logit_scale']` are newly initialized. You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.")
    return model