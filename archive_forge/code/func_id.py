import os
import unicodedata
from typing import Any, Dict, List, Optional, Tuple
import sentencepiece as sp
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging
def id(self, sym):
    logger.warning_once('The `DebertaTokenizer.id` method is deprecated and will be removed in `transformers==4.35`')
    return self.vocab[sym] if sym in self.vocab else 1