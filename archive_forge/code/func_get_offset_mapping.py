import io
import os
import unicodedata
from typing import Any, Dict, List, Optional, Tuple
import sentencepiece as spm
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging
def get_offset_mapping(self, text):
    if text is None:
        return None
    split_tokens = self.tokenize(text)
    normalized_text, char_mapping = ('', [])
    for i, ch in enumerate(text):
        if ch in self.SP_CHAR_MAPPING:
            ch = self.SP_CHAR_MAPPING.get(ch)
        else:
            ch = unicodedata.normalize('NFKC', ch)
        if self.is_whitespace(ch):
            continue
        normalized_text += ch
        char_mapping.extend([i] * len(ch))
    text, token_mapping, offset = (normalized_text, [], 0)
    if self.do_lower_case:
        text = text.lower()
    for token in split_tokens:
        if token[:1] == '▁':
            token = token[1:]
        start = text[offset:].index(token) + offset
        end = start + len(token)
        token_mapping.append((char_mapping[start], char_mapping[end - 1] + 1))
        offset = end
    return token_mapping