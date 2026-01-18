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
def set_prefix_tokens(self, language: str=None, task: str=None, predict_timestamps: bool=None):
    """
        Override the prefix tokens appended to the start of the label sequence. This method can be used standalone to
        update the prefix tokens as required when fine-tuning. Example:

        ```python
        >>> # instantiate the tokenizer and set the prefix token to Spanish
        >>> tokenizer = WhisperTokenizerFast.from_pretrained("openai/whisper-tiny", language="spanish")
        >>> # now switch the prefix token from Spanish to French
        >>> tokenizer.set_prefix_tokens(language="french")
        ```

        Args:
            language (`str`, *optional*, defaults to `None`):
                The language of the transcription text.
            task (`str`, *optional*, defaults to `None`):
                Task identifier to append at the start of sequence (if any).
            predict_timestamps (`bool`, *optional*, defaults to `None`):
                Whether to omit the `<|notimestamps|>` token at the start of the sequence.
        """
    self.language = language if language is not None else self.language
    self.task = task if task is not None else self.task
    self.predict_timestamps = predict_timestamps if predict_timestamps is not None else self.predict_timestamps
    prefix_token_ids = self.prefix_tokens
    prefixes = self.convert_ids_to_tokens(prefix_token_ids)
    eos = self.eos_token
    eos_token_id = self.eos_token_id
    prefix_template = ' '.join([f'{token}:0' for token in prefixes])
    self.backend_tokenizer.post_processor = processors.TemplateProcessing(single=f'{prefix_template} $A:0 {eos}:0', pair=f'{prefix_template} $A:0 $B:1 {eos}:1', special_tokens=[(eos, eos_token_id), *zip(prefixes, prefix_token_ids)])