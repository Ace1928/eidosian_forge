from abc import ABC, abstractmethod
from functools import lru_cache
import json
import os
import re
from typing import Dict, List, Optional, Set, Tuple
from typing_extensions import final
from parlai.core.build_data import download, make_dir
from parlai.core.opt import Opt
from parlai.utils.misc import warn_once
from parlai.utils.typing import TShared
import parlai.utils.logging as logging
def bpe_factory(opt: Opt, shared: TShared) -> 'BPEHelper':
    """
    BPE Helper Factory.

    Returns the appropriate BPE helper given the opt
    as well as available libraries.

    :param opt:
        options
    :param shared:
        shared dict

    :return BPEHelper:
        returns the appropriate BPEHelper object
    """
    from parlai.core.dict import DictionaryAgent
    tokenizer = opt.get('dict_tokenizer', DictionaryAgent.default_tok)
    bpe_helper: Optional[BPEHelper] = None
    if tokenizer == 'bytelevelbpe':
        try:
            bpe_helper = HuggingFaceBpeHelper(opt, shared)
        except ImportError:
            if opt['dict_loaded']:
                warn_once("\n\n--------------------------------------------------\n\nWARNING: You have chosen to use Huggingface's tokenizer.\nPlease install HuggingFace tokenizer with: pip install tokenizers.\nFor now, defaulting to the GPT2Tokenizer.\n\n--------------------------------------------------\n\n")
                tokenizer = 'slow_bytelevel_bpe'
            else:
                raise ImportError('Please install HuggingFace tokenizer with: pip install tokenizers.\n')
    if tokenizer == 'slow_bytelevel_bpe':
        bpe_helper = SlowBytelevelBPE(opt, shared)
    if tokenizer == 'gpt2':
        bpe_helper = Gpt2BpeHelper(opt, shared)
    if tokenizer == 'bpe':
        bpe_helper = SubwordBPEHelper(opt, shared)
    assert bpe_helper is not None, f'bpe_factory called with invalid tokenizer: {tokenizer}'
    return bpe_helper