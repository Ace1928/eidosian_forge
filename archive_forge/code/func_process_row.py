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
def process_row(self, row: List, row_index: int):
    """
        Given a row, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
    row_str = ''
    row_cell_values = []
    for cell_value in row:
        if isinstance(cell_value, int):
            row_cell_values.append(str(cell_value))
        else:
            row_cell_values.append(cell_value)
    row_str += ' | '.join(row_cell_values)
    return 'row ' + str(row_index) + ' : ' + row_str