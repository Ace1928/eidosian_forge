import json
import re
from typing import TYPE_CHECKING, List, Optional, Tuple, Union
import numpy as np
from ...utils import is_tf_available, is_torch_available, logging
from tokenizers import pre_tokenizers
from ...tokenization_utils_base import BatchEncoding
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from .tokenization_codegen import CodeGenTokenizer
def find_re(string, pattern, start_pos):
    m = pattern.search(string, start_pos)
    return m.start() if m else -1