from typing import Optional, Tuple
import torch
def codegen_wrapped_scaled_dot_product(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None):
    raise_on_head_mask(head_mask)
    batch_size = query.shape[0]
    mask_value = torch.finfo(value.dtype).min
    mask_value = torch.full([], mask_value, dtype=value.dtype)
    if batch_size == 1 and attention_mask is not None and (attention_mask[0, 0, -1, -1] < -1):
        raise ValueError("BetterTransformer does not support padding='max_length' with a batch size of 1.")
    query = query.to(value.dtype)
    key = key.to(value.dtype)
    dropout_p = self.dropout_prob_attn if self.training else 0.0
    if batch_size == 1 or self.training:
        if query.shape[2] > 1:
            sdpa_result = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=dropout_p, is_causal=True)
        else:
            sdpa_result = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=dropout_p, is_causal=False)
    else:
        query_length, key_length = (query.size(-2), key.size(-2))
        if query_length > 1:
            causal_mask = self.causal_mask[:, :, key_length - query_length:key_length, :key_length].to(torch.bool)
            causal_mask = torch.where(causal_mask, 0, mask_value)
            causal_mask = causal_mask.expand(batch_size, -1, -1, -1)
            attention_mask = torch.min(causal_mask, attention_mask)
        sdpa_result = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=dropout_p, is_causal=False)
    return (sdpa_result, None)