import os
from shutil import copyfile
from typing import List, Optional, Tuple
from tokenizers import processors
from ...tokenization_utils import AddedToken, BatchEncoding
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import is_sentencepiece_available, logging
@property
def lang_code_to_id(self):
    logger.warning_once('the `lang_code_to_id` attribute is deprecated. The logic is natively handled in the `tokenizer.adder_tokens_decoder` this attribute will be removed in `transformers` v4.38')
    return self._lang_code_to_id