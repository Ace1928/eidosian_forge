import json
import os
import re
import warnings
from functools import lru_cache
from typing import List, Optional, Tuple
import numpy as np
from tokenizers import AddedToken, pre_tokenizers, processors
from ...tokenization_utils_base import BatchEncoding
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from .english_normalizer import BasicTextNormalizer, EnglishTextNormalizer
from .tokenization_whisper import LANGUAGES, TASK_IDS, TO_LANGUAGE_CODE, WhisperTokenizer, _decode_asr
@staticmethod
def basic_normalize(text, remove_diacritics=False):
    """
        Normalize a given string using the `BasicTextNormalizer` class, which preforms commons transformation on
        multilingual text.
        """
    normalizer = BasicTextNormalizer(remove_diacritics=remove_diacritics)
    return normalizer(text)