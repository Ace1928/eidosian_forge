import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from tensorflow.compiler.tf2xla.python.xla import dynamic_update_slice
from ..modeling_tf_outputs import TFCausalLMOutputWithPast, TFSeq2SeqLMOutput
from ..models.auto import (
from ..tf_utils import shape_list, stable_softmax
from ..utils import ModelOutput, logging
from .configuration_utils import GenerationConfig
from .tf_logits_process import (
def sample_body_fn(generated, finished_sequences, cur_len, model_kwargs):
    if model_kwargs.get('past_key_values') is None or needs_full_input:
        input_ids = generated[:, :cur_len]
    else:
        input_ids = tf.expand_dims(generated[:, cur_len - 1], -1)
    model_inputs = self.prepare_inputs_for_generation(input_ids, use_cache=use_cache, **model_kwargs)
    model_outputs = self(**model_inputs, return_dict=True, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
    next_token_logits = model_outputs.logits[:, -1]
    next_tokens_scores = logits_processor(generated, next_token_logits, cur_len)
    next_tokens_scores = logits_warper(generated, next_tokens_scores, cur_len)
    if not use_xla and return_dict_in_generate:
        if output_scores:
            scores.append(next_tokens_scores)
        if output_attentions and self.config.is_encoder_decoder:
            decoder_attentions.append(model_outputs.decoder_attentions)
        elif output_attentions and (not self.config.is_encoder_decoder):
            decoder_attentions.append(model_outputs.attentions)
            if self.config.is_encoder_decoder:
                cross_attentions.append(model_outputs.cross_attentions)
        if output_hidden_states and self.config.is_encoder_decoder:
            decoder_hidden_states.append(model_outputs.decoder_hidden_states)
        elif output_hidden_states and self.config.is_encoder_decoder:
            decoder_hidden_states.append(model_outputs.hidden_states)
    if seed is not None:
        sample_seed = seed
    else:
        sample_seed = tf.experimental.numpy.random.randint(tf.int32.min, tf.int32.max, (2,), dtype=tf.int32)
    next_tokens = tf.squeeze(tf.random.stateless_categorical(logits=next_tokens_scores, num_samples=1, seed=sample_seed, dtype=tf.int32), axis=1)
    if eos_token_id is not None:
        if pad_token_id is None:
            raise ValueError('If `eos_token_id` is defined, make sure that `pad_token_id` is defined.')
        unfinished_seq = 1 - tf.cast(finished_sequences, tf.int32)
        next_tokens = next_tokens * unfinished_seq + pad_token_id * (1 - unfinished_seq)
        next_token_is_eos = tf.math.reduce_any(tf.equal(tf.broadcast_to(next_tokens, (len(eos_token_id), batch_size)), tf.expand_dims(eos_token_id, -1)), axis=0)
        finished_sequences = finished_sequences | next_token_is_eos
    update_indices = tf.stack([tf.range(batch_size), tf.broadcast_to(cur_len, [batch_size])], axis=-1)
    generated = tf.tensor_scatter_nd_update(tensor=generated, indices=update_indices, updates=next_tokens)
    cur_len += 1
    if use_xla:
        model_kwargs = self._update_model_kwargs_for_xla_generation(model_outputs=model_outputs, model_kwargs=model_kwargs, cur_len=cur_len, max_length=max_length, batch_size=batch_size, is_encoder_decoder=self.config.is_encoder_decoder, batch_axis=cache_batch_axis)
    else:
        model_kwargs = self._update_model_kwargs_for_generation(model_outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder)
        if model_kwargs.get('past_key_values', None) is None:
            model_kwargs.pop('past_key_values', None)
    return (generated, finished_sequences, cur_len, model_kwargs)