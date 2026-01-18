import collections
import os
from typing import List, Optional, Tuple
from transformers.utils import is_jieba_available, requires_backends
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging
@property
def bod_token_id(self):
    return self.encoder[self.bod_token]