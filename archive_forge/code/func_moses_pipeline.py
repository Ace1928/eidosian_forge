import glob
import os
import pickle
import re
from collections import Counter, OrderedDict
from typing import List, Optional, Tuple
import numpy as np
from ....tokenization_utils import PreTrainedTokenizer
from ....utils import (
def moses_pipeline(self, text: str) -> List[str]:
    """
        Does basic tokenization using [`sacremoses.MosesPunctNormalizer`] and [`sacremoses.MosesTokenizer`] with
        *aggressive_dash_splits=True* (see [`sacremoses.tokenize.MosesTokenizer.tokenize`]). Additionally, large
        comma-separated numbers and floating point values are split. E.g. "23,000 people are 1.80m tall" -> "23 @,@ 000
        people are 1 @.@ 80m tall"

        Args:
            text: Text to be tokenize

        Returns:
            A list of tokenized string

        Example:

        ```python
        >>> tokenizer = TransfoXLTokenizer.from_pretrained("transfo-xl/transfo-xl-wt103")
        >>> tokenizer.moses_pipeline("23,000 people are 1.80 m tall")
        ['23', '@,@', '000', 'people', 'are', '1', '@.@', '80', 'm', 'tall']
        ```"""
    text = self.moses_punct_norm(text)
    text = self.moses_tokenize(text)
    text = tokenize_numbers(text)
    return text