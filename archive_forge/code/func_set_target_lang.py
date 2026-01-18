import json
import os
import sys
import warnings
from dataclasses import dataclass
from itertools import groupby
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
import numpy as np
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import AddedToken, BatchEncoding
from ...utils import (
def set_target_lang(self, target_lang: str):
    """
        Set the target language of a nested multi-lingual dictionary
        """
    if self.vocab == self.encoder:
        raise ValueError(f'{self.vocab} is not a multi-lingual, nested tokenizer. Cannot set target language.')
    if target_lang not in self.vocab:
        raise ValueError(f'{target_lang} does not exist. Choose one of {', '.join(self.vocab.keys())}.')
    self.target_lang = target_lang
    self.init_kwargs['target_lang'] = target_lang
    self.encoder = self.vocab[target_lang]
    self.decoder = {v: k for k, v in self.encoder.items()}
    for token in self.encoder.keys():
        if len(token) > 1:
            self.add_tokens(AddedToken(token, rstrip=True, lstrip=True, normalized=False))