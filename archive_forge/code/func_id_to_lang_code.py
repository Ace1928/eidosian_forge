import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple
import sentencepiece as spm
from ...tokenization_utils import AddedToken, BatchEncoding, PreTrainedTokenizer
from ...utils import logging
@property
def id_to_lang_code(self):
    logger.warning_once('the `id_to_lang_code` attribute is deprecated. The logic is natively handled in the `tokenizer.adder_tokens_decoder` this attribute will be removed in `transformers` v4.38')
    return self._id_to_lang_code