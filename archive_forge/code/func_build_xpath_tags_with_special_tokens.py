import json
import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union
import regex as re
from ...file_utils import PaddingStrategy, TensorType, add_end_docstrings
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...tokenization_utils_base import (
from ...utils import logging
def build_xpath_tags_with_special_tokens(self, xpath_tags_0: List[int], xpath_tags_1: Optional[List[int]]=None) -> List[int]:
    pad = [self.pad_xpath_tags_seq]
    if len(xpath_tags_1) == 0:
        return pad + xpath_tags_0 + pad
    return pad + xpath_tags_0 + pad + xpath_tags_1 + pad