from typing import Optional, Tuple
import torch
def bloom_forward(self, hidden_states: torch.Tensor, residual: torch.Tensor, alibi: torch.Tensor, attention_mask: torch.Tensor, layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]]=None, head_mask: Optional[torch.Tensor]=None, use_cache: bool=False, output_attentions: bool=False, **kwargs):
    raise_on_head_mask(head_mask)
    if output_attentions is True:
        raise ValueError('output_attentions=True can not be supported with BetterTransformer.')
    fused_qkv = self.query_key_value(hidden_states)
    query_layer, key_layer, value_layer = self._split_heads(fused_qkv)
    batch_size, q_length, _, _ = query_layer.shape
    query_layer = query_layer.transpose(1, 2)
    if layer_past is not None:
        past_key, past_value = layer_past
        past_key = past_key.transpose(1, 2)
        key_layer = key_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
        value_layer = value_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
        key_layer = torch.cat((past_key, key_layer), dim=1)
        value_layer = torch.cat((past_value, value_layer), dim=1)
        key_layer = key_layer.reshape(batch_size, self.num_heads, *key_layer.shape[1:])
        value_layer = value_layer.reshape(batch_size, self.num_heads, *value_layer.shape[1:])
    else:
        key_layer = key_layer.transpose(1, 2)
        value_layer = value_layer.transpose(1, 2)
    alibi = alibi.reshape(batch_size, -1, *alibi.shape[1:])
    alibi = torch.masked_fill(alibi, attention_mask, torch.finfo(alibi.dtype).min)
    context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer, attn_mask=alibi, dropout_p=self.dropout_prob_attn if self.training else 0.0)
    context_layer = context_layer.transpose(1, 2)
    context_layer = context_layer.reshape(*context_layer.shape[:2], -1)
    if self.pretraining_tp > 1 and self.slow_but_exact:
        slices = self.hidden_size / self.pretraining_tp
        output_tensor = torch.zeros_like(context_layer)
        for i in range(self.pretraining_tp):
            output_tensor = output_tensor + torch.nn.functional.linear(context_layer[:, :, int(i * slices):int((i + 1) * slices)], self.dense.weight[:, int(i * slices):int((i + 1) * slices)])
    else:
        output_tensor = self.dense(context_layer)
    output_tensor = torch.nn.functional.dropout(output_tensor, p=self.hidden_dropout, training=self.training)
    output_tensor = residual + output_tensor
    if use_cache is True:
        present = (key_layer.reshape(-1, *key_layer.shape[2:]).transpose(1, 2), value_layer.reshape(-1, *value_layer.shape[2:]))
    else:
        present = None
    return (output_tensor, present)