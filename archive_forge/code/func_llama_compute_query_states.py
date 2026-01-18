import inspect
import torch
import torch.nn as nn
def llama_compute_query_states(model: nn.Module, **kwargs) -> torch.Tensor:
    """
    Compute query states for Llama models specifically. They need to be recomputed as the forward() method of the
    original LlamaModel in the transformers library does not return them. See the related discussion in the PR:
    https://github.com/huggingface/peft/pull/268
    """
    hidden_states = kwargs.get('hidden_states')
    position_ids = kwargs.get('position_ids')
    past_key_value = kwargs.get('past_key_value')
    bsz, q_len, _ = hidden_states.size()
    query_states = model.q_proj(hidden_states).view(bsz, q_len, model.num_heads, model.head_dim).transpose(1, 2)
    value_states = model.v_proj(hidden_states).view(bsz, q_len, model.num_heads, model.head_dim).transpose(1, 2)
    seq_len = q_len
    if past_key_value is not None:
        if isinstance(past_key_value, tuple):
            seq_len += past_key_value[0].shape[-2]
        else:
            seq_len += past_key_value.get_seq_length(model.layer_idx)
    if 'position_ids' not in inspect.signature(model.rotary_emb.forward).parameters:
        cos, sin = model.rotary_emb(value_states, seq_len=seq_len)
        return llama_apply_rotary_pos_emb(query_states, cos, sin, position_ids)
    past_seen_tokens = 0
    if position_ids is None:
        if past_key_value is None:
            new_cache_positions = torch.arange(q_len, q_len + q_len, device=value_states.device)
        else:
            past_seen_tokens = past_key_value.get_usable_length(q_len, model.layer_idx)
            new_cache_positions = torch.arange(past_seen_tokens, past_seen_tokens + q_len, device=value_states.device)
        position_ids = new_cache_positions.unsqueeze(0)
    cos, sin = model.rotary_emb(value_states, seq_len=q_len + past_seen_tokens, position_ids=position_ids)
    if len(cos.shape) == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    return query_states * cos + llama_rotate_half(query_states) * sin