import re
from functools import partial
from multiprocessing import Pool
from typing import List, Union
import numpy as np
from transformers.tokenization_utils_base import INIT_TOKENIZER_DOCSTRING
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.utils import add_end_docstrings
from ...utils import is_levenshtein_available, is_nltk_available, logging, requires_backends
def normalize_list_like_lines(generation):
    """
    Normalize lines in the given text that resemble list items. The function looks for lines that start optionally with
    '-' or '*', possibly followed by Roman numerals or digits indicating nesting levels. The function reformats such
    lines to make them more structured.

    Args:
        generation (str): The input text containing lines that need to be normalized.

    Returns:
        str: The input text with the list-like lines normalized.

    Note:
        The function uses regular expressions to identify and reformat the list-like lines. The patterns capture
        optional bullet points, nesting levels indicated by numerals, and the actual list item content. The
        normalization adjusts the bullet point style and nesting levels based on the captured patterns.
    """
    pattern = '(?:^)(-|\\*)?(?!-|\\*) ?((?:\\d|[ixv])+ )?.+? (-|\\*) (((?:\\d|[ixv])+)\\.(\\d|[ixv]) )?.*(?:$)'
    for match in reversed(list(re.finditer(pattern, generation, flags=re.I | re.M))):
        start, stop = match.span()
        delim = match.group(3) + ' '
        splits = match.group(0).split(delim)
        replacement = ''
        if match.group(1) is not None:
            splits = splits[1:]
            delim1 = match.group(1) + ' '
        else:
            delim1 = ''
            continue
        pre, post = (generation[:start], generation[stop:])
        for i, item in enumerate(splits):
            level = 0
            potential_numeral, _, rest = item.strip().partition(' ')
            if not rest:
                continue
            if re.match('^[\\dixv]+((?:\\.[\\dixv])?)+$', potential_numeral, flags=re.I | re.M):
                level = potential_numeral.count('.')
            replacement += ('\n' if i > 0 else '') + '\t' * level + (delim if i > 0 or start == 0 else delim1) + item.strip()
        if post == '':
            post = '\n'
        generation = pre + replacement + post
    return generation