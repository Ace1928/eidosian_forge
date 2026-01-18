import os
import re
import warnings
from shutil import copyfile
from typing import List, Optional, Tuple
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import is_sentencepiece_available, logging
def get_sentinel_tokens(self):
    return list(set(filter(lambda x: bool(re.search('<extra_id_\\d+>', x)) is not None, self.additional_special_tokens)))