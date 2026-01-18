import re
from functools import partial
from multiprocessing import Pool
from typing import List, Union
import numpy as np
from transformers.tokenization_utils_base import INIT_TOKENIZER_DOCSTRING
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.utils import add_end_docstrings
from ...utils import is_levenshtein_available, is_nltk_available, logging, requires_backends
def get_slices(lines, clean_lines):
    """
    Get slices of text based on specific criteria within the lines.

    This function identifies and returns slices of text from the input lines based on certain conditions.

    These conditions were chosen by the Nougat authors:
    - The slice is less than 200 characters long.
    - The slice is more than 3 characters long.
    - The slice does not start with "[MISSING_PAGE".
    - The slice is either the same as the next slice or the ratio of the two in terms of Levensthein distance is
      greater than 0.9.

    Args:
        lines (`List[str]`):
            The list of lines containing the text.
        clean_lines (`List[str]`):
            A cleaned version of the text (without numbers).

    Returns:
        `List[tuple]`: A list of tuples representing the start and end indices of text slices.
    """
    indices = np.zeros(len(lines))
    for i in range(len(lines) - 1):
        j = i + 1
        while not clean_lines[j] and j < len(lines) - 1:
            j += 1
        if len(clean_lines[i]) < 200 and len(clean_lines[i]) > 3 and (len(clean_lines[j]) < 200) and (len(clean_lines[j]) > 3) and (not clean_lines[i].startswith('[MISSING_PAGE')) and (clean_lines[i] == clean_lines[j] or ratio(clean_lines[i], clean_lines[j]) > 0.9):
            indices[i:j] = 1
    ids = np.where(indices)[0]
    slices = []
    if len(ids) == 0:
        return slices
    j0 = 0
    for j, x in enumerate(np.diff(ids) > 3):
        if x:
            slices.append((ids[j0], ids[j] + 2))
            j0 = j + 1
    slices.append((ids[j0], ids[-1] + 2))
    return [sli for sli in slices if sli[1] - sli[0] > 15]