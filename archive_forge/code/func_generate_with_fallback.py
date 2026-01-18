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
def generate_with_fallback(self, segment_input, decoder_input_ids, cur_bsz, batch_idx_map, seek, num_segment_frames, max_frames, temperatures, generation_config, logits_processor, stopping_criteria, prefix_allowed_tokens_fn, synced_gpus, return_token_timestamps, do_condition_on_prev_tokens, kwargs):
    seek_sequence_list = [None for _ in range(cur_bsz)]
    seek_outputs_list = [None for _ in range(cur_bsz)]
    needs_fallback = [False for _ in range(cur_bsz)]
    should_skip = [False for _ in range(cur_bsz)]
    fallback_index_map = list(range(cur_bsz))
    if generation_config.no_speech_threshold is not None:
        self._setup_no_speech_detection(logits_processor, segment_input, decoder_input_ids, kwargs)
    for fallback_idx, temperature in enumerate(temperatures):
        generation_config.do_sample = temperature is not None and temperature > 0.0
        generation_config.temperature = temperature if generation_config.do_sample else 1.0
        generation_config.num_beams = kwargs.pop('num_beams', 1) if not generation_config.do_sample else 1
        seek_outputs = super().generate(segment_input, generation_config, logits_processor, stopping_criteria, prefix_allowed_tokens_fn, synced_gpus, decoder_input_ids=decoder_input_ids, **kwargs)
        seek_sequences, seek_outputs = self._postprocess_outputs(seek_outputs=seek_outputs, decoder_input_ids=decoder_input_ids, return_token_timestamps=return_token_timestamps, generation_config=generation_config)
        new_fallback_index_map = []
        new_segment_input = []
        new_decoder_input_ids = []
        new_decoder_attention_mask = []
        for i, seek_sequence in enumerate(seek_sequences):
            prev_i = batch_idx_map[fallback_index_map[i]]
            is_not_final = seek[prev_i] + num_segment_frames < max_frames[prev_i]
            if is_not_final and seek_sequence[-1] == generation_config.eos_token_id:
                seek_sequence = seek_sequence[:-1]
            if seek_sequence[-1] == generation_config.pad_token_id:
                num_paddings = (seek_sequence == generation_config.pad_token_id).sum()
                seek_sequence = seek_sequence[:-num_paddings]
            needs_fallback[i], should_skip[i] = self._need_fallback(seek_sequence, seek_outputs, i, logits_processor, generation_config, self.config.vocab_size, temperature)
            seek_sequence_list[fallback_index_map[i]] = seek_sequence
            seek_outputs_list[fallback_index_map[i]] = seek_outputs[i]
            is_low_temperature = temperature is None or temperature < 0.5
            do_condition_on_prev_tokens[fallback_index_map[i]] = generation_config.condition_on_prev_tokens and is_low_temperature
            if needs_fallback[i]:
                new_fallback_index_map.append(fallback_index_map[i])
                new_segment_input.append(segment_input[i])
                new_decoder_input_ids.append(decoder_input_ids[i])
                if 'decoder_attention_mask' in kwargs:
                    new_decoder_attention_mask.append(kwargs['decoder_attention_mask'][i])
        fallback_index_map = new_fallback_index_map
        if len(fallback_index_map) == 0 or fallback_idx == len(temperatures) - 1:
            seek_sequences = seek_sequence_list
            seek_outputs = seek_outputs_list
            break
        decoder_input_ids = torch.stack(new_decoder_input_ids)
        segment_input = torch.stack(new_segment_input)
        if 'decoder_attention_mask' in kwargs:
            kwargs['decoder_attention_mask'] = torch.stack(new_decoder_attention_mask)
    return (seek_sequences, seek_outputs, should_skip, do_condition_on_prev_tokens)