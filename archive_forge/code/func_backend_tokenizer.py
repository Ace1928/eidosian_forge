import copy
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union
import tokenizers.pre_tokenizers as pre_tokenizers_fast
from tokenizers import Encoding as EncodingFast
from tokenizers import Tokenizer as TokenizerFast
from tokenizers.decoders import Decoder as DecoderFast
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordLevelTrainer, WordPieceTrainer
from .convert_slow_tokenizer import convert_slow_tokenizer
from .tokenization_utils import PreTrainedTokenizer
from .tokenization_utils_base import (
from .utils import PaddingStrategy, add_end_docstrings, logging
@property
def backend_tokenizer(self) -> TokenizerFast:
    """
        `tokenizers.implementations.BaseTokenizer`: The Rust tokenizer used as a backend.
        """
    return self._tokenizer