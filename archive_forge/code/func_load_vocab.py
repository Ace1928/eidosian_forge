import collections
import os
import unicodedata
from typing import List, Optional, Tuple
from ....tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ....utils import logging
def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, 'r', encoding='utf-8') as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip('\n')
        vocab[token] = index
    return vocab