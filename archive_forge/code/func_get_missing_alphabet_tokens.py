import os
import warnings
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from multiprocessing import Pool, get_context, get_start_method
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Union
import numpy as np
from ...processing_utils import ProcessorMixin
from ...utils import ModelOutput, logging, requires_backends
@staticmethod
def get_missing_alphabet_tokens(decoder, tokenizer):
    from pyctcdecode.alphabet import BLANK_TOKEN_PTN, UNK_TOKEN, UNK_TOKEN_PTN
    tokenizer_vocab_list = list(tokenizer.get_vocab().keys())
    for i, token in enumerate(tokenizer_vocab_list):
        if BLANK_TOKEN_PTN.match(token):
            tokenizer_vocab_list[i] = ''
        if token == tokenizer.word_delimiter_token:
            tokenizer_vocab_list[i] = ' '
        if UNK_TOKEN_PTN.match(token):
            tokenizer_vocab_list[i] = UNK_TOKEN
    missing_tokens = set(tokenizer_vocab_list) - set(decoder._alphabet.labels)
    return missing_tokens