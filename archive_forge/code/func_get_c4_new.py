import random
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from datasets import load_dataset
def get_c4_new(tokenizer: Any, seqlen: int, nsamples: int, split: str='train'):
    if split == 'train':
        data = load_dataset('allenai/c4', split='train', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'})
    elif split == 'validation':
        data = load_dataset('allenai/c4', split='validation', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'})
    dataset = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(data) - 1)
            enc = tokenizer(data[i]['text'], return_tensors='pt')
            if enc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = enc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        dataset.append({'input_ids': inp, 'attention_mask': attention_mask})
    return dataset