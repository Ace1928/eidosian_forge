import json
import os
import random
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union
import regex as re
from ....file_utils import ExplicitEnum, PaddingStrategy, TensorType, add_end_docstrings, is_pandas_available
from ....tokenization_utils import AddedToken, PreTrainedTokenizer
from ....tokenization_utils_base import ENCODE_KWARGS_DOCSTRING, BatchEncoding, TextInput, TruncationStrategy
from ....utils import logging
def estimate_delete_ratio(self, table_content: Dict, question: str, max_length=None):
    if 'header' not in table_content or 'rows' not in table_content:
        raise ValueError("The table content should contain both 'header' and 'rows' keys.")
    question_tokens = self.tokenize(question, add_special_tokens=True)
    header_string = self.table_linearize.process_header(table_content['header'])
    header_tokens = self.tokenize(header_string, add_special_tokens=False)
    used_token_len = len(question_tokens) + len(header_tokens)
    remain_token_len = max_length - used_token_len
    value_string = ''
    for _, row_example in enumerate(table_content['rows']):
        value_string += self.table_linearize.process_row(row_example, 100) + ' '
    value_token_len = len(self.tokenize(value_string))
    if value_token_len < remain_token_len:
        return (0.0, remain_token_len)
    else:
        return (1.0 - remain_token_len / value_token_len, remain_token_len)