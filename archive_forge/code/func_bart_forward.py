from typing import Optional, Tuple
import torch
def bart_forward(self, hidden_states: torch.Tensor, key_value_states: Optional[torch.Tensor]=None, past_key_value: Optional[Tuple[torch.Tensor]]=None, attention_mask: Optional[torch.Tensor]=None, layer_head_mask: Optional[torch.Tensor]=None, output_attentions: bool=False, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel"""
    raise_on_head_mask(layer_head_mask)
    if output_attentions is True:
        raise ValueError('output_attentions=True can not be supported with BetterTransformer.')
    is_cross_attention = key_value_states is not None
    bsz, tgt_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    if is_cross_attention and past_key_value is not None and (past_key_value[0].shape[2] == key_value_states.shape[1]):
        key_states = past_key_value[0]
        value_states = past_key_value[1]
    elif is_cross_attention:
        key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
        value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    elif past_key_value is not None:
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
    else:
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    if self.is_decoder:
        past_key_value = (key_states, value_states)
    query_states = self._shape(query_states, tgt_len, bsz)
    key_states = key_states
    value_states = value_states
    attn_output = torch.nn.functional.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=attention_mask, dropout_p=self.dropout if self.training else 0.0, is_causal=False)
    if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):
        raise ValueError(f'`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}')
    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    attn_output = self.out_proj(attn_output)
    return (attn_output, None, past_key_value)