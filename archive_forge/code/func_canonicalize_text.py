import os
import re
import string
import warnings
from shutil import copyfile
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
import sentencepiece as spm
from ...convert_slow_tokenizer import import_protobuf
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import AddedToken
from ...utils import logging, requires_backends
def canonicalize_text(self, text, *, keep_punctuation_exact_string=None):
    """Returns canonicalized `text` (puncuation removed).

        Args:
            text (`str`):
                String to be canonicalized.
            keep_punctuation_exact_string (`str`, *optional*):
                If provided, then this exact string is kept. For example providing '{}' will keep any occurrences of '{}'
                (but will still remove '{' and '}' that appear separately).
        """
    if keep_punctuation_exact_string:
        text = keep_punctuation_exact_string.join((self.remove_punctuation(part) for part in text.split(keep_punctuation_exact_string)))
    else:
        text = self.remove_punctuation(text)
    text = re.sub('\\s+', ' ', text)
    text = text.strip()
    return text