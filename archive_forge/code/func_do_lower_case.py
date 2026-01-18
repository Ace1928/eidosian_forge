import collections
import os
import unicodedata
from typing import List, Optional, Tuple
from ....tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ....utils import logging
@property
def do_lower_case(self):
    return self.basic_tokenizer.do_lower_case