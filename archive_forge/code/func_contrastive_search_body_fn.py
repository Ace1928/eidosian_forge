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
def contrastive_search_body_fn(generated, finished_sequences, cur_len, model_kwargs, next_step_cached_variables):
    """state update fn."""
    if model_kwargs.get('past_key_values') is None:
        model_inputs = self.prepare_inputs_for_generation(generated[:, :cur_len], use_cache=use_cache, **model_kwargs)
        outputs = self(**model_inputs, return_dict=True, output_hidden_states=True, output_attentions=output_attentions)
        if self.config.is_encoder_decoder:
            last_hidden_states = outputs.decoder_hidden_states[-1]
        else:
            last_hidden_states = outputs.hidden_states[-1]
        if use_xla:
            last_hidden_states = tf.pad(last_hidden_states, [[0, 0], [0, max_length - cur_len], [0, 0]])
        logit_for_next_step = outputs.logits[:, -1, :]
        if use_xla:
            model_kwargs = self._update_model_kwargs_for_xla_generation(model_outputs=outputs, model_kwargs=model_kwargs, cur_len=cur_len, max_length=max_length, batch_size=batch_size, is_encoder_decoder=self.config.is_encoder_decoder, batch_axis=cache_batch_axis)
        else:
            model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder)
        _, model_kwargs = self._expand_inputs_for_generation(expand_size=top_k, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs)
        past_key_values = model_kwargs.get('past_key_values')
        if past_key_values is None:
            raise ValueError(f"{self.__class__.__name__} does not support caching and therefore **can't** be used for contrastive search.")
        elif not isinstance(past_key_values[0], (tuple, tf.Tensor)) or past_key_values[0][0].shape[0] != batch_size:
            raise ValueError(f"{self.__class__.__name__} does not have a standard cache format and therefore **can't** be used for contrastive search without further modifications.")
    else:
        logit_for_next_step = next_step_cached_variables['logit_for_next_step']
        last_hidden_states = next_step_cached_variables['last_hidden_states']
        outputs = next_step_cached_variables['outputs']
    logit_for_next_step = logits_processor(generated, logit_for_next_step, cur_len)
    logit_for_next_step = logits_warper(generated, logit_for_next_step, cur_len)
    next_probs = stable_softmax(logit_for_next_step, axis=-1)
    top_k_probs, top_k_ids = tf.math.top_k(next_probs, k=top_k)
    if not use_xla and return_dict_in_generate:
        if output_scores:
            scores.append(logit_for_next_step)
        if output_attentions and self.config.is_encoder_decoder:
            decoder_attentions.append(outputs.decoder_attentions)
        elif output_attentions and (not self.config.is_encoder_decoder):
            decoder_attentions.append(outputs.attentions)
            if self.config.is_encoder_decoder:
                cross_attentions.append(outputs.cross_attentions)
        if output_hidden_states and self.config.is_encoder_decoder:
            decoder_hidden_states.append(outputs.decoder_hidden_states)
        elif output_hidden_states and self.config.is_encoder_decoder:
            decoder_hidden_states.append(outputs.hidden_states)
    model_kwargs['past_key_values'] = tf.nest.map_structure(lambda tensor: tf.repeat(tensor, top_k, axis=cache_batch_axis), model_kwargs['past_key_values'])
    next_model_inputs = self.prepare_inputs_for_generation(tf.reshape(top_k_ids, [-1, 1]), use_cache=use_cache, **model_kwargs)
    outputs = self(**next_model_inputs, return_dict=True, output_hidden_states=True, output_attentions=output_attentions)
    next_past_key_values = self._extract_past_from_model_output(outputs)
    logits = outputs.logits[:, -1, :]
    if self.config.is_encoder_decoder:
        next_hidden = outputs.decoder_hidden_states[-1]
        full_hidden_states = outputs.decoder_hidden_states
    else:
        next_hidden = outputs.hidden_states[-1]
        full_hidden_states = outputs.hidden_states
    context_hidden = tf.repeat(last_hidden_states[:, :cur_len, :], top_k, axis=0)
    selected_idx = _ranking_fast(context_hidden, next_hidden, top_k_probs, penalty_alpha, top_k)
    selected_idx_stacked = selected_idx + tf.range(selected_idx.shape[0], dtype=tf.int64) * top_k
    next_tokens = tf.gather(top_k_ids, selected_idx, axis=1, batch_dims=1)
    next_hidden = gather_best_candidate(next_hidden, selected_idx_stacked)
    if use_xla:
        last_hidden_states = dynamic_update_slice(last_hidden_states, next_hidden, [0, cur_len, 0])
    else:
        last_hidden_states = tf.concat([last_hidden_states, next_hidden], axis=1)
    next_decoder_hidden_states = gather_best_candidate(full_hidden_states, selected_idx_stacked)
    next_past_key_values = gather_best_candidate(next_past_key_values, selected_idx_stacked, batch_axis=cache_batch_axis)
    logit_for_next_step = gather_best_candidate(logits, selected_idx_stacked)
    if self.config.is_encoder_decoder:
        next_step_cross_attentions = ()
        next_step_decoder_attentions = ()
        if output_attentions:
            next_step_cross_attentions = gather_best_candidate(outputs.cross_attentions, selected_idx_stacked)
            next_step_decoder_attentions = gather_best_candidate(outputs.decoder_attentions, selected_idx_stacked)
        outputs = TFSeq2SeqLMOutput(past_key_values=next_past_key_values, decoder_hidden_states=next_decoder_hidden_states, decoder_attentions=next_step_decoder_attentions or None, cross_attentions=next_step_cross_attentions or None)
    else:
        next_step_attentions = ()
        if output_attentions:
            next_step_attentions = gather_best_candidate(outputs.attentions, selected_idx_stacked)
        outputs = TFCausalLMOutputWithPast(past_key_values=next_past_key_values, hidden_states=next_decoder_hidden_states, attentions=next_step_attentions or None)
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
        model_kwargs = self._update_model_kwargs_for_xla_generation(model_outputs=outputs, model_kwargs=model_kwargs, cur_len=cur_len + 1, max_length=max_length, batch_size=batch_size * top_k, is_encoder_decoder=self.config.is_encoder_decoder, batch_axis=cache_batch_axis)
    else:
        model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder)
    next_step_cached_variables = {'logit_for_next_step': logit_for_next_step, 'last_hidden_states': last_hidden_states, 'outputs': outputs}
    return (generated, finished_sequences, cur_len, model_kwargs, next_step_cached_variables)