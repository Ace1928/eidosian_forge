import gc
import random
import warnings
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import top_k_top_p_filtering
from .import_utils import is_npu_available, is_xpu_available
def respond_to_batch(model: nn.Module, queries: List[torch.LongTensor], txt_len: int=20, top_k: int=0, top_p: float=1.0) -> torch.LongTensor:
    """Sample text from language model."""
    input_ids = queries
    for _i in range(txt_len):
        outputs = model(input_ids)
        next_token_logits = outputs[0][:, -1, :]
        next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
    return input_ids[:, -txt_len:]