import io
import os
import unicodedata
from typing import Any, Dict, List, Optional, Tuple
import sentencepiece as spm
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging
def build_offset_mapping_with_special_tokens(self, offset_mapping_0, offset_mapping_1=None):
    """
        Build offset map from a pair of offset map by concatenating and adding offsets of special tokens. An Ernie-M
        offset_mapping has the following format:

        - single sequence: `(0,0) X (0,0)`
        - pair of sequences: `(0,0) A (0,0) (0,0) B (0,0)`

        Args:
            offset_mapping_ids_0 (`List[tuple]`):
                List of char offsets to which the special tokens will be added.
            offset_mapping_ids_1 (`List[tuple]`, *optional*):
                Optional second list of wordpiece offsets for offset mapping pairs.
        Returns:
            `List[tuple]`: List of wordpiece offsets with the appropriate offsets of special tokens.
        """
    if offset_mapping_1 is None:
        return [(0, 0)] + offset_mapping_0 + [(0, 0)]
    return [(0, 0)] + offset_mapping_0 + [(0, 0), (0, 0)] + offset_mapping_1 + [(0, 0)]