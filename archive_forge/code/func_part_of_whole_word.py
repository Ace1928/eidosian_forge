import os
import unicodedata
from typing import Any, Dict, List, Optional, Tuple
import sentencepiece as sp
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging
def part_of_whole_word(self, token, is_bos=False):
    logger.warning_once('The `DebertaTokenizer.part_of_whole_word` method is deprecated and will be removed in `transformers==4.35`')
    if is_bos:
        return True
    if len(token) == 1 and (_is_whitespace(list(token)[0]) or _is_control(list(token)[0]) or _is_punctuation(list(token)[0])) or token in self.special_tokens:
        return False
    word_start = b'\xe2\x96\x81'.decode('utf-8')
    return not token.startswith(word_start)