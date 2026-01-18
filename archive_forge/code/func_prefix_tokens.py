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
@property
def prefix_tokens(self) -> List[int]:
    bos_token_id = self.convert_tokens_to_ids('<|startoftranscript|>')
    translate_token_id = self.convert_tokens_to_ids('<|translate|>')
    transcribe_token_id = self.convert_tokens_to_ids('<|transcribe|>')
    notimestamps_token_id = self.convert_tokens_to_ids('<|notimestamps|>')
    langs = tuple(LANGUAGES.keys())
    if self.language is not None:
        self.language = self.language.lower()
        if self.language in TO_LANGUAGE_CODE:
            language_id = TO_LANGUAGE_CODE[self.language]
        elif self.language in TO_LANGUAGE_CODE.values():
            language_id = self.language
        else:
            is_language_code = len(self.language) == 2
            raise ValueError(f'Unsupported language: {self.language}. Language should be one of: {(list(TO_LANGUAGE_CODE.values()) if is_language_code else list(TO_LANGUAGE_CODE.keys()))}.')
    if self.task is not None:
        if self.task not in TASK_IDS:
            raise ValueError(f'Unsupported task: {self.task}. Task should be in: {TASK_IDS}')
    bos_sequence = [bos_token_id]
    if self.language is not None:
        bos_sequence.append(bos_token_id + 1 + langs.index(language_id))
    if self.task is not None:
        bos_sequence.append(transcribe_token_id if self.task == 'transcribe' else translate_token_id)
    if not self.predict_timestamps:
        bos_sequence.append(notimestamps_token_id)
    return bos_sequence