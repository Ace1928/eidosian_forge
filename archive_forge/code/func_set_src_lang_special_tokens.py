import os
from shutil import copyfile
from typing import List, Optional, Tuple, Union
from tokenizers import processors
from ...tokenization_utils import (
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import PaddingStrategy, is_sentencepiece_available, logging
def set_src_lang_special_tokens(self, src_lang) -> None:
    """Reset the special tokens to the source lang setting.
        Prefix=[src_lang_code], suffix = [eos]
        """
    self.cur_lang_code = self.convert_tokens_to_ids(src_lang)
    if self.cur_lang_code == self.unk_token_id:
        logger.warning_once(f'`tgt_lang={src_lang}` has not be found in the `vocabulary`. Behaviour will probably be unexpected because the language token id will be replaced by the unknown token id.')
    self.init_kwargs['src_lang'] = src_lang
    self.prefix_tokens = [self.cur_lang_code]
    self.suffix_tokens = [self.eos_token_id]
    prefix_tokens_str = self.convert_ids_to_tokens(self.prefix_tokens)
    suffix_tokens_str = self.convert_ids_to_tokens(self.suffix_tokens)
    self._tokenizer.post_processor = processors.TemplateProcessing(single=prefix_tokens_str + ['$A'] + suffix_tokens_str, pair=prefix_tokens_str + ['$A', '$B'] + suffix_tokens_str, special_tokens=list(zip(prefix_tokens_str + suffix_tokens_str, self.prefix_tokens + self.suffix_tokens)))