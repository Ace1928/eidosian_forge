import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import is_phonemizer_available, logging
def normalize_text(self, input_string):
    """Lowercase the input string, respecting any special token ids that may be part or entirely upper-cased."""
    all_vocabulary = list(self.encoder.keys()) + list(self.added_tokens_encoder.keys())
    filtered_text = ''
    i = 0
    while i < len(input_string):
        found_match = False
        for word in all_vocabulary:
            if input_string[i:i + len(word)] == word:
                filtered_text += word
                i += len(word)
                found_match = True
                break
        if not found_match:
            filtered_text += input_string[i].lower()
            i += 1
    return filtered_text