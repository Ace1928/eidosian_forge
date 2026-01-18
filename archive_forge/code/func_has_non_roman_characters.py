import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import is_phonemizer_available, logging
def has_non_roman_characters(input_string):
    non_roman_pattern = re.compile('[^\\x00-\\x7F]')
    match = non_roman_pattern.search(input_string)
    has_non_roman = match is not None
    return has_non_roman