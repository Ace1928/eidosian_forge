import dataclasses
import math
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Tuple
import torch
def sequence_generator(model: Callable, sampler: Callable, fsms: List['Guide'], token_ids: torch.Tensor, sequence_weights: torch.Tensor, attention_masks: torch.Tensor, fsm_states: List[int], rng: torch.Generator=torch.Generator()) -> Iterator[GenerationState]:
    """Generates sequences of tokens.

    Parameters
    ----------
    model
        A callable that generates a probability distribution over the
        vocabulary when passed a tensor of token ids.
    sampler
        A callable that returns the next token ids, their ancestor sequence and
        the updated sequence weights when passed a distribution over the
        vocabulary.
    token_ids
        A tensor of token ids on which the sequence distribution is conditioned, of
        shape ``(n_seqs, n_prompt_tokens)``
    sequence_weights
        A tensor that contains the initial weights of the sequences, of shape
        ``(n_seqs,)``
    attention_masks
        A tensor of tensors that represent the tokens considered at the attention
        layer, of shape ``(n_seqs, n_prompt_tokens)``.
    fsms
        List of finite-state machines that drive the text generation,
        one for each sequence in the batch.
    fsm_states
        The initial states of the finite-state machine for each sequence in the batch.

    Yields
    ------
    A new sequence.

    """
    kv_cache = None
    while True:
        try:
            logits, kv_cache = model(token_ids, attention_masks, kv_cache)
        except IndexError:
            raise ContextLengthExceededError('The input length exceeds the context length of the model.')
        allowed_tokens = get_allowed_tokens(fsms, fsm_states)
        biased_logits = bias_logits(logits, allowed_tokens)
        next_token_ids, ancestors, sequence_weights = sampler(biased_logits, sequence_weights, rng)
        token_ids = update_token_ids(token_ids, next_token_ids, ancestors)
        attention_masks = update_attention_masks(attention_masks, ancestors)
        kv_cache = reorder_kv_cache(kv_cache, ancestors)
        fsms = reorder_fsms(fsms, ancestors)
        fsm_states = reorder_fsm_states(fsm_states, ancestors)
        fsm_states = get_next_fsm_states(fsms, fsm_states, next_token_ids)
        is_finished = is_generation_finished(fsms, fsm_states)
        if is_finished:
            yield GenerationState(token_ids, kv_cache, logits, sequence_weights, fsm_states)
            return
        yield GenerationState(token_ids, kv_cache, logits, sequence_weights, fsm_states)