import random
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from datasets import load_dataset
def get_ptb_new(tokenizer: Any, seqlen: int, nsamples: int, split: str='train'):
    if split == 'train':
        data = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    elif split == 'validation':
        data = load_dataset('ptb_text_only', 'penn_treebank', split='test')
    enc = tokenizer(' '.join(data['sentence']), return_tensors='pt')
    dataset = []
    for _ in range(nsamples):
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = enc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        dataset.append({'input_ids': inp, 'attention_mask': attention_mask})
    return dataset