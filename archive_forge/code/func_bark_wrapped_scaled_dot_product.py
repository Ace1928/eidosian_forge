from typing import Optional, Tuple
import torch
def bark_wrapped_scaled_dot_product(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None):
    raise_on_head_mask(head_mask)
    is_causal = self.is_causal and query.shape[2] != 1
    sdpa_result = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal)
    return (sdpa_result, None)