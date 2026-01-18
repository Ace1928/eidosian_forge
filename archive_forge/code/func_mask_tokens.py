import random
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from random import randint
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import numpy as np
from ..models.bert import BertTokenizer, BertTokenizerFast
from ..tokenization_utils_base import PreTrainedTokenizerBase
from ..utils import PaddingStrategy
def mask_tokens(self, inputs: Any) -> Tuple[Any, Any, Any]:
    """
        Prepare masked tokens inputs/labels/attention_mask for masked language modeling: 80% MASK, 10% random, 10%
        original. N-gram not applied yet.
        """
    import torch
    if self.tokenizer.mask_token is None:
        raise ValueError('This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer.')
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, self.mlm_probability)
    special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if self.tokenizer._pad_token is not None:
        padding_mask = labels.eq(self.tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    attention_mask = (~masked_indices).float()
    if self.tokenizer._pad_token is not None:
        attention_padding_mask = labels.eq(self.tokenizer.pad_token_id)
        attention_mask.masked_fill_(attention_padding_mask, value=1.0)
    labels[~masked_indices] = -100
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]
    return (inputs, labels, attention_mask)