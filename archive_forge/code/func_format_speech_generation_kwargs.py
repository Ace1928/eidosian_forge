import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_seamless_m4t import SeamlessM4TConfig
def format_speech_generation_kwargs(kwargs):
    """
    Format kwargs for SeamlessM4T models that generate speech, attribute kwargs to either the text generation or the
    speech generation models.

    Args:
        kwargs (`dict`)`:
             Keyword arguments are of two types:

                - Without a prefix, they will be entered as `**kwargs` for the `generate` method of each sub-model,
                except for `decoder_input_ids` which will only be passed through the text components.
                - With a *text_* or *speech_* prefix, they will be input for the `generate` method of the
                text model and speech model respectively. It has the priority over the keywords without a prefix.

                This means you can, for example, specify a generation strategy for one generation but not for the
                other.
    """
    kwargs_text = {}
    kwargs_speech = {}
    for key, value in kwargs.items():
        if key.startswith('text_'):
            key = key[len('text_'):]
            kwargs_text[key] = value
        elif key.startswith('speech_'):
            key = key[len('speech_'):]
            kwargs_speech[key] = value
        else:
            if key not in kwargs_text:
                kwargs_text[key] = value
            if key not in kwargs_speech:
                kwargs_speech[key] = value
    return (kwargs_text, kwargs_speech)