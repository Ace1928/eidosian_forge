import os
from shutil import copyfile
from typing import List, Optional, Tuple, Union
from tokenizers import processors
from ...tokenization_utils import (
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import PaddingStrategy, is_sentencepiece_available, logging
@src_lang.setter
def src_lang(self, new_src_lang: str) -> None:
    if '__' not in new_src_lang:
        self._src_lang = f'__{new_src_lang}__'
    else:
        self._src_lang = new_src_lang
    self.set_src_lang_special_tokens(self._src_lang)