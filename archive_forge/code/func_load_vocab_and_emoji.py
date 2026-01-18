import collections
import json
import os
import re
from typing import Optional, Tuple
import numpy as np
from ...tokenization_utils_fast import PreTrainedTokenizer
from ...utils import logging
def load_vocab_and_emoji(vocab_file, emoji_file):
    """Loads a vocabulary file and emoji file into a dictionary."""
    with open(emoji_file, 'r', encoding='utf-8') as f:
        emoji = json.loads(f.read())
    vocab = collections.OrderedDict()
    raw_vocab = collections.OrderedDict()
    ids_to_tokens = collections.OrderedDict()
    with open(vocab_file, 'r', encoding='utf-8') as f:
        token = f.readlines()
    token = [[t.rstrip('\n')] if t == ',' or ',' not in t else t.rstrip('\n').split(',') for t in token]
    for idx, b in enumerate(token):
        ids_to_tokens[idx] = b
        raw_vocab[','.join(b)] = idx
        for wd in b:
            vocab[wd] = idx
    return (vocab, raw_vocab, ids_to_tokens, emoji)