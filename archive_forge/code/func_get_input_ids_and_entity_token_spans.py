import itertools
import json
import os
from collections.abc import Mapping
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import regex as re
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import (
from ...utils import add_end_docstrings, is_tf_tensor, is_torch_tensor, logging
def get_input_ids_and_entity_token_spans(text, entity_spans):
    if entity_spans is None:
        return (get_input_ids(text), None)
    cur = 0
    input_ids = []
    entity_token_spans = [None] * len(entity_spans)
    split_char_positions = sorted(frozenset(itertools.chain(*entity_spans)))
    char_pos2token_pos = {}
    for split_char_position in split_char_positions:
        orig_split_char_position = split_char_position
        if split_char_position > 0 and text[split_char_position - 1] == ' ':
            split_char_position -= 1
        if cur != split_char_position:
            input_ids += get_input_ids(text[cur:split_char_position])
            cur = split_char_position
        char_pos2token_pos[orig_split_char_position] = len(input_ids)
    input_ids += get_input_ids(text[cur:])
    entity_token_spans = [(char_pos2token_pos[char_start], char_pos2token_pos[char_end]) for char_start, char_end in entity_spans]
    return (input_ids, entity_token_spans)