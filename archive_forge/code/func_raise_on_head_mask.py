from typing import Optional, Tuple
import torch
def raise_on_head_mask(head_mask: Optional[torch.Tensor]):
    if head_mask is not None:
        raise ValueError('layer_head_mask different than None is unsupported for now with BetterTransformer, pleaseopen a PR or an issue at https://github.com/huggingface/optimum.')