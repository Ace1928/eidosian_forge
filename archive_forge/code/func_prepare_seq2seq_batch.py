import os
from shutil import copyfile
from typing import List, Optional, Tuple, Union
from tokenizers import processors
from ...tokenization_utils import (
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import PaddingStrategy, is_sentencepiece_available, logging
def prepare_seq2seq_batch(self, src_texts: List[str], src_lang: str='eng', tgt_texts: Optional[List[str]]=None, tgt_lang: str='fra', **kwargs) -> BatchEncoding:
    self.src_lang = src_lang
    self.tgt_lang = tgt_lang
    return super().prepare_seq2seq_batch(src_texts, tgt_texts, **kwargs)