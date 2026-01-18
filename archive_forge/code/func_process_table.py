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
def process_table(self, table_content: Dict):
    """
        Given a table, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
    assert 'header' in table_content and 'rows' in table_content, self.PROMPT_MESSAGE
    table_str = self.process_header(table_content['header']) + ' '
    for i, row_example in enumerate(table_content['rows']):
        table_str += self.process_row(row_example, row_index=i + 1) + ' '
    return table_str.strip()