import copy
import math
import warnings
import zlib
from typing import Callable, Iterator, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from ...generation.configuration_utils import GenerationConfig
from ...generation.logits_process import (
from ...generation.stopping_criteria import StoppingCriteriaList
from ...modeling_outputs import BaseModelOutput
from ...utils import logging
from .tokenization_whisper import TASK_IDS, TO_LANGUAGE_CODE
def detect_language(self, input_features: Optional[torch.FloatTensor]=None, encoder_outputs: Optional[Union[torch.FloatTensor, BaseModelOutput]]=None, generation_config: Optional[GenerationConfig]=None, num_segment_frames: int=3000) -> torch.Tensor:
    """
        Detects language from log-mel input features or encoder_outputs

        Parameters:
            input_features (`torch.Tensor` of shape `(batch_size, feature_size, sequence_length)`, *optional*):
                Float values of log-mel features extracted from the raw speech waveform. The raw speech waveform can be obtained by
                loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via
                the soundfile library (`pip install soundfile`). To prepare the array into `input_features`, the
                [`AutoFeatureExtractor`] should be used for extracting the mel features, padding and conversion into a
                tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`] for details.
            encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
                Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
                `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
                hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            num_segment_frames (`int`, defaults to 3000):
                The number of log-mel frames the model expects

        Return:
            A `torch.LongTensor` representing the detected language ids.
        """
    if input_features is None and encoder_outputs is None:
        raise ValueError('You have to specify either `input_features` or `encoder_outputs`')
    elif input_features is not None and encoder_outputs is not None:
        raise ValueError('Make sure to specificy only one of `input_features` or `encoder_outputs` - not both!')
    elif input_features is not None:
        inputs = {'input_features': input_features[:, :, :num_segment_frames]}
        batch_size = input_features.shape[0]
    elif encoder_outputs is not None:
        inputs = {'encoder_outputs': encoder_outputs}
        batch_size = encoder_outputs[0].shape[0] if isinstance(encoder_outputs, BaseModelOutput) else encoder_outputs[0]
    generation_config = generation_config or self.generation_config
    decoder_input_ids = torch.ones((batch_size, 1), device=self.device, dtype=torch.long) * generation_config.decoder_start_token_id
    with torch.no_grad():
        logits = self(**inputs, decoder_input_ids=decoder_input_ids).logits[:, -1]
    non_lang_mask = torch.ones_like(logits[0], dtype=torch.bool)
    non_lang_mask[list(generation_config.lang_to_id.values())] = False
    logits[:, non_lang_mask] = -np.inf
    lang_ids = logits.argmax(-1)
    return lang_ids